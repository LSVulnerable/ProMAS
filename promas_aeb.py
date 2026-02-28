import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
import glob
from tqdm import tqdm
import random
import torch.nn.functional as F
import math
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

# Set Deterministic Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class CausalFeedbackModel(nn.Module):
    """
    Causal Feedback Model (Refactored):
    - RESPONSIBLE for extracting Logical Connection Features.
    - TRAINS Feature Projection to separate Success/Failure states.
    - EXTRACTS Delta Features (Change in State).
    """
    def __init__(self, base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", n_clusters=30):
        super().__init__()
        
        # 1. Resolve Model Path
        self.model_path = base_model_name
        possible_paths = [
            # "/home/ls/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots",
            "/home/ls/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots",
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ]
        
        for root_path in possible_paths:
            if os.path.exists(root_path) and os.path.isdir(root_path):
                subdirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
                if subdirs:
                    candidate = subdirs[0] 
                    if os.path.exists(os.path.join(candidate, "config.json")):
                        self.model_path = candidate
                        print(f"Loaded local Llama backbone: {self.model_path}")
                        break
        
        # 2. Initialize Components
        print(f"Initializing Tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" 
        
        print(f"Initializing Backbone from {self.model_path}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="cuda:0"
        )
        
        # Freeze Backbone completely
        print("Freezing Llama Backbone...")
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        hidden_size = self.base_model.config.hidden_size # 4096
        self.execution_device = self.base_model.device 
        self.target_dtype = self.base_model.dtype
        
        # 3. Feature Projection / Attention Pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        ).to(self.execution_device).to(self.target_dtype)
        
        # Learnable Projection (Will be trained)
        # Unified Delta Projection (Start is projected here too)
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024)
        ).to(self.execution_device).to(self.target_dtype)
        
        # 4. Proactive Prediction Heads
        # Unified Head: Proactive Head (History/Context -> Next Cluster)
        self.proactive_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, n_clusters) 
        ).to(self.execution_device).to(self.target_dtype)

    def forward(self, input_ids, attention_mask, return_pooled=False):
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_4 = torch.stack(hidden_states[-4:], dim=0) 
            combined_hidden = torch.mean(last_4, dim=0).to(self.execution_device).to(self.target_dtype)
        
        # Attention Pooling
        attn_weights = self.attention_pooling(combined_hidden) 
        mask_expanded = attention_mask.unsqueeze(-1).to(self.execution_device)
        attn_weights = attn_weights * mask_expanded
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        pooled_embedding = torch.sum(combined_hidden * attn_weights, dim=1)
        
        if return_pooled:
            return pooled_embedding
            
        # Project to causal feature space
        features = self.feature_projection(pooled_embedding)
        features = F.normalize(features, p=2, dim=1)
        
        return features

    def extract_feature(self, task_context, prev_feedback, curr_action, prev_vector=None):
        """
        Calculates Current Vector AND Delta.
        Returns: (delta_vector, current_vector)
        """
        prompt = f"[TASK]\n{task_context}\n\n[PREVIOUS_FEEDBACK]\n{prev_feedback}\n\n[CURRENT_ACTION]\n{curr_action}"
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=8192, truncation=True, padding=True
        ).to(self.execution_device)
        
        current_vector = self.forward(inputs.input_ids, inputs.attention_mask, return_pooled=True) # (1, 1024)

        if prev_vector is None:
            # First step: Delta is just the Current Vector (Implicitly Delta from Zero)
            # OR project the Absolute Vector using separate logic?
            # Reverting to Unified: Project the Absolute Vector as if it were a Delta
            raw_delta = current_vector
        else:
            # Step > 0: Delta Features (Current - Previous)
            raw_delta = current_vector - prev_vector.to(self.execution_device)
        
        delta_projected = self.feature_projection(raw_delta)
        delta_projected = F.normalize(delta_projected, p=2, dim=1)
        
        return delta_projected, current_vector

    def predict_next_cluster_probs(self, task_context, history_text, is_start=False):
        """
        Proactive Prediction: Given History, predicts the Probability Distribution of the NEXT action cluster.
        """
        # We need a representation of the 'Current State' before the action happens.
        # Use the history text to get a 'State Vector'.
        prompt = f"[TASK]\n{task_context}\n\n[HISTORY]\n{history_text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=8192, truncation=True, padding=True).to(self.execution_device)
        
        # Get State Embedding
        with torch.no_grad():
            state_emb = self.forward(inputs.input_ids, inputs.attention_mask, return_pooled=True)
            # Use RAW state embedding (4096) directly
            
        # Predict Next Cluster Distribution using UNIFIED HEAD
        logits = self.proactive_head(state_emb) 
        probs = F.softmax(logits, dim=1)
        
        return probs

    def train_proactive_head(self, dataset, markov_model):
        """
        Step C: Train the Proactive Head to classify Next Action Cluster.
        Uses CrossEntropyLoss against K-Means labels.
        UNIFIED: Train single head on all data (Start + History).
        """
        print("\nTraining Proactive Prediction Heads (Classification)...")
        # Unpack Kmeans (now single)
        kmeans = markov_model.kmeans
        
        self.proactive_head.requires_grad_(True)
        self.attention_pooling.requires_grad_(True) # Enable training for pooling
        self.feature_projection.eval()
        
        # Add attention_pooling to optimizer
        params = list(self.proactive_head.parameters()) + list(self.attention_pooling.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print("Collecting History->Cluster Pairs for Training...")
        limit_files = min(len(dataset), 1000) 
        
        all_embeddings = []
        all_labels = []
        
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            
            if mistake_step == -1: continue
            limit = min(len(history), mistake_step + 1)
            
            prev_emb = None
            
            for i in range(limit):
                # --- TARGET GENERATION (Y) ---
                curr_msg = history[i]
                ground_truth_action_text = f"{curr_msg.get('name', 'Unknown')}: {curr_msg.get('content')}"
                
                # --- INPUT GENERATION (X) ---
                if i == 0: 
                    # Step 0: Input is pure Task Context.
                    history_text_for_input = task_context
                    prev_feedback_for_target = task_context 
                else: 
                    # Step > 0: Input is History up to i-1.
                    past_msgs = history[max(0, i-1):i] 
                    history_text_for_input = "\n".join([f"{m.get('name')}: {m.get('content')}" for m in past_msgs])
                    
                    prev_msg = history[i-1]
                    prev_feedback_for_target = f"{prev_msg.get('name')}: {prev_msg.get('content')}"
                
                # 1. Forward Pass (INPUT -> STATE)
                p_prompt = f"[TASK]\n{task_context}\n\n[HISTORY]\n{history_text_for_input}"
                p_inp = self.tokenizer(p_prompt, return_tensors="pt", max_length=8192, truncation=True).to(self.execution_device)
                with torch.no_grad():
                    state_emb = self.forward(p_inp.input_ids, p_inp.attention_mask, return_pooled=True)
                
                # 2. Label Generation (TARGET -> CLUSTER ID)
                t_prompt = f"[TASK]\n{task_context}\n\n[PREVIOUS_FEEDBACK]\n{prev_feedback_for_target}\n\n[CURRENT_ACTION]\n{ground_truth_action_text}"
                t_inp = self.tokenizer(t_prompt, return_tensors="pt", max_length=8192, truncation=True).to(self.execution_device)
                with torch.no_grad():
                    curr_emb = self.forward(t_inp.input_ids, t_inp.attention_mask, return_pooled=True)
                
                if prev_emb is None:
                    raw_delta = curr_emb 
                else:
                    raw_delta = curr_emb - prev_emb 
                
                prev_emb = curr_emb

                # Project ground truth & Get Label
                target_delta = self.feature_projection(raw_delta)
                target_delta = F.normalize(target_delta, p=2, dim=1)
                
                label = kmeans.predict(target_delta.detach().float().cpu().numpy())[0]
                
                all_embeddings.append(state_emb.cpu())
                all_labels.append(label)

        
        batch_size = 32
        epochs = 15
        
        # --- TRAIN UNIFIED HEAD ---
        if all_embeddings:
            X_train = torch.cat(all_embeddings, dim=0) # (N, 4096)
            Y_train = torch.tensor(all_labels, dtype=torch.long) # (N,)
            print(f"Training Unified Prediction Head ({len(X_train)} samples)...")
            
            for epoch in range(epochs):
                self.proactive_head.train()
                total_loss = 0
                steps = 0
                perm = torch.randperm(X_train.size(0))
                
                for i in range(0, X_train.size(0), batch_size):
                    idx = perm[i:i+batch_size]
                    bx = X_train[idx].to(self.execution_device).to(self.target_dtype)
                    by = Y_train[idx].to(self.execution_device)
                    
                    optimizer.zero_grad()
                    logits = self.proactive_head(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    steps += 1
                    
                print(f"  Head Epoch {epoch}: Loss = {total_loss/steps:.4f}")

        self.proactive_head.requires_grad_(False)
        self.attention_pooling.requires_grad_(False)

    def train_projection_layer(self, dataset):
        """
        Step B: Contrastive Training for Projection Layer (Triplet Loss).
        Unified: Single Loop for all deltas (Start & Action).
        """
        print("\nTraining Projection Layer (Contrastive / Triplet)...")
        self.feature_projection.requires_grad_(True)
        self.attention_pooling.requires_grad_(True) # Enable training for pooling
        
        # Optimizer: Include Attention Pooling
        params = list(self.feature_projection.parameters()) + list(self.attention_pooling.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        
        criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=1.0
        )
        
        # 1. Collect Data
        failure_deltas = []
        hard_negative_deltas = [] # T-1 steps
        success_deltas = [] # Random successes
        
        # Note: We now treat Start Vectors as part of the same latent space contrastive task
        # so they will be mixed in if they are part of a failure/success chain. 
        # But specifically, we need to handle the "raw delta" logic carefully.
        
        print("Collecting Raw Deltas for Training (with Hard Negatives)...")
        limit_files = min(len(dataset), 500) 
        
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            
            limit = len(history)
            
            prev_emb = None
            prev_raw_delta = None

            for i in range(limit):
                curr_msg = history[i]
                curr_action = f"{curr_msg.get('name', 'Unknown')}: {curr_msg.get('content')}"
                if i == 0: prev_feedback = task_context
                else: prev_feedback = f"{history[i-1].get('name')}: {history[i-1].get('content')}"
                
                # We need the RAW 4096-dim embedding before projection
                prompt = f"[TASK]\n{task_context}\n\n[PREVIOUS_FEEDBACK]\n{prev_feedback}\n\n[CURRENT_ACTION]\n{curr_action}"
                inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(self.execution_device)
                
                with torch.no_grad():
                    # Get the pooled output (4096 dim)
                    curr_emb = self.forward(inp.input_ids, inp.attention_mask, return_pooled=True) 
                
                if prev_emb is None:
                    # Step 0: Raw Delta is Current
                    raw_delta = curr_emb 
                else:
                    # Step > 0: Relative Delta
                    raw_delta = curr_emb - prev_emb
                
                # Store for Triplet Training (Only if mistake defined)
                if mistake_step != -1:
                    if i == mistake_step:
                        failure_deltas.append(raw_delta.cpu())
                        # Hard Negative: Previous Step Delta
                        if prev_raw_delta is not None:
                            hard_negative_deltas.append(prev_raw_delta.cpu())
                    elif i < mistake_step: # Only learn from pre-mistake successes
                            success_deltas.append(raw_delta.cpu())
                
                prev_emb = curr_emb
                # Update prev_raw_delta (used for hard negatives)
                prev_raw_delta = raw_delta

        if not failure_deltas:
            print("Insufficient training data.")
            return

        print(f"Collected: {len(failure_deltas)} Fail Deltas")
        
        # Convert to tensors
        X_fail = torch.cat(failure_deltas, dim=0)
        X_hard = torch.cat(hard_negative_deltas, dim=0) if hard_negative_deltas else X_fail 
        # Fallback if no hard negatives (e.g. all failures at step 0)
        
        min_len = min(len(X_fail), len(X_hard))
        X_fail = X_fail[:min_len]
        X_hard = X_hard[:min_len]
        
        X_succ_rand = torch.cat(success_deltas, dim=0) if success_deltas else X_fail # Fallback
        
        # 2. Train Loop
        batch_size = 32
        epochs = 15
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.feature_projection.train()
            epoch_loss = 0
            steps = 0
            
            n_samples = X_fail.size(0)
            perm = torch.randperm(n_samples)
            if n_samples > 0:
                for i in range(0, n_samples, batch_size):
                    idx = perm[i:i+batch_size]
                    batch_anchor_raw = X_fail[idx].to(self.execution_device).to(self.target_dtype)
                    
                    # Positive: Random other Failures (Cluster all failures together)
                    ridx_pos = torch.randint(0, X_fail.size(0), (batch_anchor_raw.size(0),))
                    batch_pos_raw = X_fail[ridx_pos].to(self.execution_device).to(self.target_dtype)
                    
                    # Negative 1: Hard Negative
                    batch_neg_hard_raw = X_hard[idx].to(self.execution_device).to(self.target_dtype)
                    
                    # Negative 2: Random Success
                    ridx_neg = torch.randint(0, X_succ_rand.size(0), (batch_anchor_raw.size(0),))
                    batch_neg_rand_raw = X_succ_rand[ridx_neg].to(self.execution_device).to(self.target_dtype)
                    
                    optimizer.zero_grad()
                    
                    anc = F.normalize(self.feature_projection(batch_anchor_raw), p=2, dim=1)
                    pos = F.normalize(self.feature_projection(batch_pos_raw), p=2, dim=1)
                    neg_hard = F.normalize(self.feature_projection(batch_neg_hard_raw), p=2, dim=1)
                    neg_rand = F.normalize(self.feature_projection(batch_neg_rand_raw), p=2, dim=1)
                    
                    loss = criterion(anc, pos, neg_hard) + 0.5 * criterion(anc, pos, neg_rand)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    steps += 1
            
            print(f"Epoch {epoch+1}: Loss = {epoch_loss/(steps+1e-9):.4f}")
            
        print("Training Complete. Freezing Projection Layers.")
        self.feature_projection.requires_grad_(False)
        self.attention_pooling.requires_grad_(False)


class VectorMarkovEntropy:
    """
    Refactored Vector Markov Entropy (Discrete):
    - Quantizes Deltas to Discrete Actions (KMeans).
    - DUELING KMEANS: Separates Start Vectors from Delta Vectors.
    - Builds Transition Probabilities (Success/Fail Matrices).
    """
    def __init__(self, device='cpu', n_clusters=30):
        self.kmeans = None
        self.n_clusters = n_clusters
        self.device = device
        
        # State Transitions (From Action Cluster -> To Action Cluster)
        self.count_fail_matrix = None
        self.count_succ_matrix = None
        
        self.start_fail = None 
        self.start_succ = None 
        
        self.threshold = 0.0
        
    def get_state_id(self, cluster_id):
        # Identity mapping since output of predict IS the state id
        return cluster_id

    def fit_quantization(self, dataset, model):
        """
        Step A: Extract Vectors and Fit Unified KMeans Model.
        """
        print("\nStep A: Quantizing Action Space (Unified KMeans)...")
        all_vectors = []
        
        limit_files = min(len(dataset), 1000)
        
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            limit = len(history)
            task_context = json_data.get('question', '')
            prev_vector = None
            
            for i in range(limit):
                curr_msg = history[i]
                curr_action = f"{curr_msg.get('name')}: {curr_msg.get('content')}"
                if i == 0: prev_feedback = task_context
                else: prev_feedback = f"{history[i-1].get('name')}: {history[i-1].get('content')}"
                
                # Extract
                delta_proj, curr_vector = model.extract_feature(task_context, prev_feedback, curr_action, prev_vector)
                
                vec_np = delta_proj.detach().float().cpu().numpy()
                all_vectors.append(vec_np)
                    
                prev_vector = curr_vector
                    
        # Fit Unified KMeans
        if all_vectors:
            sv = np.vstack(all_vectors)
            print(f"Fitting Unified KMeans on {len(sv)} vectors...")
            self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=SEED, batch_size=2048)
            self.kmeans.fit(sv)
        
    def build_markov_matrices(self, dataset, model):
        """
        Step B: Build Transition Matrices.
        - Unified: Single Matrix (N x N) tracking Count(Prev -> Next).
        """
        print("\nStep B: Building Markov Transition Matrices...")
        
        # Initialize Matrices
        self.count_fail_matrix = np.zeros((self.n_clusters, self.n_clusters))
        self.count_succ_matrix = np.zeros((self.n_clusters, self.n_clusters))        
        
        self.start_fail = np.zeros(self.n_clusters)
        self.start_succ = np.zeros(self.n_clusters)
        
        limit_files = min(len(dataset), 1000)
        
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            
            if mistake_step == -1: continue 
            limit = min(len(history), mistake_step + 1)
            
            prev_vector = None
            prev_state = -1 
            
            for i in range(limit):
                curr_msg = history[i]
                curr_action = f"{curr_msg.get('name')}: {curr_msg.get('content')}"
                if i == 0: prev_feedback = task_context
                else: prev_feedback = f"{history[i-1].get('name')}: {history[i-1].get('content')}"
                
                delta_proj, curr_vector = model.extract_feature(task_context, prev_feedback, curr_action, prev_vector)
                
                # Quantize using UNIFIED KMeans
                vec_np = delta_proj.detach().float().cpu().numpy()
                curr_cluster = self.kmeans.predict(vec_np)[0]
                
                curr_state = curr_cluster 
                
                is_fail_path = (i == mistake_step) 
                
                if prev_state != -1:
                    # Transition: Prev -> Curr
                    if is_fail_path:
                        self.count_fail_matrix[prev_state, curr_state] += 1
                    else:
                        self.count_succ_matrix[prev_state, curr_state] += 1
                else:
                    # i == 0. Start -> Curr
                    if is_fail_path:
                        self.start_fail[curr_state] += 1
                    else:
                        self.start_succ[curr_state] += 1
                
                prev_vector = curr_vector
                prev_state = curr_state

        # APPLY SMOOTHING (平衡平滑参数)
        self.count_fail_matrix += 0.5
        self.count_succ_matrix += 0.5
        self.start_fail += 0.5
        self.start_succ += 0.5
        
        print("Markov Matrices Built (Unified Space).")
        
    def calculate_proactive_risk(self, prev_state_id, cluster_probs, is_start=False):
        """
        Calculate Risk.
        Args:
            prev_state_id: Previous cluster ID. (Ignored if is_start=True)
            cluster_probs: Predicted distribution.
            is_start: Whether we are predicting Step 0.
        """
        if self.kmeans is None: return 0.0, -1
        
        probs = cluster_probs.detach().float().cpu().numpy().flatten()
        
        next_cluster_id = np.argmax(probs)
        next_state_id = next_cluster_id 
        
        if is_start:
            # Use Start Statistics
            c_fail = self.start_fail
            c_succ = self.start_succ
        else:
            if prev_state_id == -1: 
                 c_fail = self.count_fail_matrix.sum(axis=0) 
                 c_succ = self.count_succ_matrix.sum(axis=0)
            elif prev_state_id < self.n_clusters: 
                 c_fail = self.count_fail_matrix[prev_state_id]
                 c_succ = self.count_succ_matrix[prev_state_id]
            else:
                 return 0.0, next_state_id

        total_counts = c_fail + c_succ + 1e-9
        failure_likelihood = (c_fail) / (total_counts + 1)

        # Top-K Risk (Sharpened)
        top_k_indices = np.argsort(probs)[-5:] # Look at top 5
        top_probs = probs[top_k_indices]
        top_probs = top_probs / (np.sum(top_probs) + 1e-9)
        top_failures = failure_likelihood[top_k_indices]
        
        risk = np.sum(top_probs * top_failures)
        
        return risk, next_state_id



    def calibrate_threshold(self, dataset, model):
        """
        Step D: Calibrate Risk Threshold using KMeans on Risk Distribution.
        """
        print("\nStep D: Calibrating Risk Threshold (KMeans)...")
        risks = []
        
        limit_files = min(len(dataset), 500) 
        
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            task_context = json_data.get('question', '')
            
            prev_actual_vector = None
            prev_actual_state = -1
            
            for i in range(len(history)):
                curr_msg = history[i]
                
                # History Context for PREDICTION
                if i == 0: 
                    history_text = task_context
                    prev_feedback = task_context
                else: 
                    start_h = max(0, i-5)
                    context_msgs = history[start_h:i]
                    history_text = "\n".join([f"Step {j}: [{m.get('name')}] {m.get('content')[:500]}" for j, m in enumerate(context_msgs, start_h)])
                    prev_feedback = f"{history[i-1].get('name')}: {history[i-1].get('content')}"

                # 1. Prediction
                with torch.no_grad():
                    cluster_probs = model.predict_next_cluster_probs(task_context, history_text, is_start=(i==0))
                
                # 2. Risk
                risk, _ = self.calculate_proactive_risk(prev_actual_state, cluster_probs, is_start=(i==0))
                risks.append(risk)

                # 3. Update State (Ground Truth)
                curr_agent = curr_msg.get('name', 'Unknown')
                curr_action = f"{curr_agent}: {curr_msg.get('content')}"
                
                with torch.no_grad():
                    actual_delta, curr_vector = model.extract_feature(task_context, prev_feedback, curr_action, prev_actual_vector)
                
                prev_actual_vector = curr_vector
                
                # Quantize with UNIFIED model
                vec_np = actual_delta.detach().float().cpu().numpy()
                actual_cluster = self.kmeans.predict(vec_np)[0]
                prev_actual_state = self.get_state_id(actual_cluster)

        
        if not risks: 
            self.threshold = 0.02
            return
        
        # Cluster Risks into 2 groups: Low Risk (Safe) vs High Risk (Potential Fail)
        X = np.array(risks).reshape(-1, 1)
        # Using MiniBatchKMeans from sklearn.cluster
        km = MiniBatchKMeans(n_clusters=2, random_state=SEED, batch_size=2048).fit(X)
        centers = sorted(km.cluster_centers_.flatten())
        
        # Threshold is the decision boundary (midpoint)
        # Or we can be more conservative: e.g. 75% between low and high
        self.threshold = (centers[0] + centers[1]) / 2
        
        print(f"Calibrated Threshold: {self.threshold:.4f} (Centers: {centers[0]:.4f}, {centers[1]:.4f})")

class WhoWhenDataset:
    def __init__(self, directory_path=None, file_paths=None):
        if file_paths:
            self.file_paths = file_paths
        elif directory_path:
            self.file_paths = glob.glob(os.path.join(directory_path, "*.json"))
        else:
            raise ValueError("Must provide either directory_path or file_paths")
        print(f"Initialized dataset with {len(self.file_paths)} files.")

    def load_file(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx): return self.load_file(self.file_paths[idx])

class AgentErrorBench:
    def __init__(self, directory_path=None, file_paths=None):
        self.data_entries = []
        paths = []
        if file_paths:
            if isinstance(file_paths, str):
                paths = [file_paths]
            else:
                paths = file_paths
        elif directory_path:
            paths = glob.glob(os.path.join(directory_path, "*.json"))
        
        for p in paths:
            self._load_data(p)
            
        print(f"Initialized AgentErrorBench with {len(self.data_entries)} entries from {len(paths)} files.")

    def _load_data(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try as single list/dict
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    self.data_entries.extend(data)
                    return
                elif isinstance(data, dict):
                    self.data_entries.append(data)
                    return
            except json.JSONDecodeError:
                pass
            
            # Try as concatenated JSON
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                while pos < len(content) and content[pos].isspace():
                    pos += 1
                if pos >= len(content):
                    break
                try:
                    obj, end = decoder.raw_decode(content, idx=pos)
                    self.data_entries.append(obj)
                    pos = end
                except json.JSONDecodeError:
                    break
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def __len__(self): 
        return len(self.data_entries)

    def __getitem__(self, idx): 
        entry = self.data_entries[idx]
        
        # Parse full_trajectory string
        traj_str = entry.get("full_trajectory", "{}")
        history = []
        try:
            traj_json = json.loads(traj_str)
            history = traj_json.get("messages", [])
            # Only keep dicts, sometimes messages can be weird
            if not isinstance(history, list):
                history = []
        except:
            history = []
        
        # Map fields
        # critical_failure_step -> mistake_step
        mistake_step = entry.get("critical_failure_step", -1)
        
        # Get question
        question = ""
        if history and isinstance(history[0], dict):
            question = history[0].get("content", "")
            
        return {
            "history": history,
            "mistake_step": mistake_step,
            "question": question,
            "original_id": entry.get("trajectory_id")
        }

# --- Main Evaluation Loop ---

def run_causal_evaluation(model, vector_markov, dataset):
    """
    Refactored Evaluation Loop: PROACTIVE
    - At step t, uses History[0..t-1] and Next_Agent_ID.
    - Predicts Delta_{t} (Future Action).
    - Checks Risk of Transition (State_{t-1} -> Predicted_State_{t}).
    - Updates State using ACTUAL Delta_{t} after observation.
    """
    print("\n--- Running Proactive Risk Simulation ---")
    
    # Initialize statistics dictionary
    stats = {
        "total_mistakes": 0,
        "detected": 0,
        "agent_hits": 0,
        "early_detection": 0
    }
    
    # Risk Threshold
    # Use calibrated threshold if available
    RISK_THRESHOLD = vector_markov.threshold if hasattr(vector_markov, 'threshold') else 0.02
    print(f"Using Risk Threshold: {RISK_THRESHOLD:.3f}")

    model.eval()

    real_ratio = []
    detect_ratio = []
    
    for idx, json_data in enumerate(tqdm(dataset)):
        history = json_data.get('history', [])
        ms = json_data.get('mistake_step', -1)
        mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
        task_context = json_data.get('question', '')
        
        if mistake_step == -1: continue
            
        target_agent = history[mistake_step].get('name', history[mistake_step].get('role', 'Unknown'))
        stats["total_mistakes"] += 1
        limit = min(len(history), mistake_step + 1)
        
        detected_step = -1
        risk_trace = []

        real_ratio.append(mistake_step / len(history))
        
        # We need to track the ACTUAL state path to query the Markov matrix correctly
        prev_actual_vector = None
        prev_actual_state = -1 
        
        first_detection_triggered = False 
        outcome = "MISSED"
        agent_hit_str = ""
        
        for i in range(limit):
            curr_msg = history[i]
            curr_agent = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
            # We predict BEFORE seeing action
            
            # History Context for PREDICTION
            if i == 0: 
                history_text = task_context
                prev_feedback = task_context
            else: 
                # 扩展上下文窗口到5步
                start_h = max(0, i-5)
                context_msgs = history[start_h:i]
                history_text = "\n".join([f"Step {j}: [{m.get('name')}] {m.get('content')[:500]}" for j, m in enumerate(context_msgs, start_h)])
                prev_feedback = f"{history[i-1].get('name')}: {history[i-1].get('content')}"

            # --- 1. PROACTIVE PREDICTION ---
            with torch.no_grad():
                cluster_probs = model.predict_next_cluster_probs(task_context, history_text, is_start=(i==0))
            
            # --- 2. RISK CALCULATION ---
            # Risk = P(Fail | Prev_State -> Predicted_Next_State) ...
            risk_val, _ = vector_markov.calculate_proactive_risk(prev_actual_state, cluster_probs, is_start=(i==0))
            risk_trace.append(risk_val)
            
            # --- 3. OBSERVATION & STATE UPDATE (Ground Truth) ---
            # Now we allow ourselves to see what ACTUALLY happened to update our state for the NEXT prediction
            curr_action = f"{curr_agent}: {curr_msg.get('content')}"
            
            with torch.no_grad():
                actual_delta, curr_vector = model.extract_feature(task_context, prev_feedback, curr_action, prev_actual_vector)
            
            prev_actual_vector = curr_vector
            
            # Use UNIFIED KMeans
            vec_np = actual_delta.detach().float().cpu().numpy()
            actual_cluster = vector_markov.kmeans.predict(vec_np)[0]
                
            prev_actual_state = vector_markov.get_state_id(actual_cluster)

            # --- 4. Detection Logic (Risk Jump Strategy) ---
            # To reduce Early Stops (False Positives at Step 0/1), we look for sudden increasing risk.
            # Stable high risk might just mean "Ambiguous Cluster", but a JUMP means "Situation Deteriorated".
            
            prev_risk_val = risk_trace[-2] if len(risk_trace) >= 2 else 0.0
            risk_jump = risk_val - prev_risk_val
            
            # 计算累积风险 (指数衰减加权)
            cumulative_risk = 0.0
            decay = 0.75  # 减少衰减系数，让近期风险更重要
            if risk_trace:
                weights = [decay ** (len(risk_trace) - 1 - j) for j in range(len(risk_trace))]
                cumulative_risk = sum(w * r for w, r in zip(weights, risk_trace)) / (sum(weights) + 1e-9)
            
            # 动态阈值：随步骤推进逐渐降低（后期更敏感），但前期保持较高
            progress_ratio = i / max(limit, 1)
            # 前 30% 步骤保持较高阈值，之后逐渐降低
            if progress_ratio < 0.3:
                dynamic_threshold = RISK_THRESHOLD * 1.1  # 前期提高阈值减少误报
            else:
                dynamic_threshold = RISK_THRESHOLD * (1 - (progress_ratio - 0.3) * 0.35)
            
            should_trigger = False
            trigger_score = 0  # 触发分数累计
            
            # Rule 1: 累积风险超过阈值且当前风险较高 (+2分)
            if cumulative_risk > dynamic_threshold * 0.9 and risk_val > dynamic_threshold * 0.85:
                trigger_score += 2
            
            # Rule 2: 连续3步风险上升 (+2分)
            if len(risk_trace) >= 3:
                recent = risk_trace[-3:]
                if recent[0] < recent[1] < recent[2] and risk_val > dynamic_threshold:
                    trigger_score += 2
                
            # Rule 3: 显著风险跳跃 (+2分)
            if risk_val > dynamic_threshold and risk_jump > 0.15:
                trigger_score += 2
            
            # Rule 4: 高置信度直接触发 (+3分)
            if risk_val > 0.6:
                trigger_score += 3
            
            # Rule 5: 中等风险持续多步 (+1分)
            if len(risk_trace) >= 4:
                recent_avg = sum(risk_trace[-4:]) / 4
                if recent_avg > dynamic_threshold * 0.9:
                    trigger_score += 1
            
            # 需要累计达到2分才触发（多条件联合）
            if trigger_score >= 2:
                should_trigger = True
            
            if should_trigger:
                first_detection_triggered = True
                detected_step = i
            
            # --- End Detection Logic ---

            
            if first_detection_triggered:
                # first_detection_triggered = True
                # detected_step = i
                
                agent_hit_str = "AGENT UNMATCH"
                if target_agent == curr_agent or (target_agent != 'Unknown' and target_agent in curr_agent):
                    stats["agent_hits"] += 1
                    agent_hit_str = "AGENT MATCH"
                
                if detected_step == mistake_step:
                    stats["detected"] += 1
                    outcome = "EXACT_HIT"
                elif detected_step < mistake_step:
                    stats["early_detection"] += 1
                    outcome = "EARLY_STOP"
                else:
                    outcome = "LATE"
                break

        debug_info = f"Max Risk: {max(risk_trace):.2f}" if risk_trace else ""
        
        if first_detection_triggered:
            tqdm.write(f"[ID {idx}] {outcome} @ {detected_step} (Tgt: {mistake_step}) | {agent_hit_str} | {debug_info}")
            detect_ratio.append(detected_step / len(history))
        else:
            tqdm.write(f"[ID {idx}] MISSED {mistake_step} | {debug_info}")
            detect_ratio.append(1.0)

    print("\n=== Final Results ===")
    print(f"Total Cases: {stats['total_mistakes']}")
    print(f"Step Hit: {stats['detected']} ({stats['detected']/stats['total_mistakes']:.2%})")
    print(f"Agent Hit: {stats['agent_hits']} ({stats['agent_hits']/stats['total_mistakes']:.2%})")
    print(f"Early Warning: {stats['early_detection']}")
    print(f"Missed: {stats['total_mistakes'] - stats['detected'] - stats['early_detection']}")
    print(f"Avg Real Mistake Ratio: {np.mean(real_ratio):.4f}")
    print(f"Avg Detected Mistake Ratio: {np.mean(detect_ratio) if detect_ratio else 0.0:.4f}")
    
    # # Calculate AUC-ROC
    # if len(all_step_labels) > 0:
    #     try:
    #         auc_score = roc_auc_score(all_step_labels, all_step_risks)
    #         print(f"Step-wise Risk AUC-ROC: {auc_score:.4f}")
    #     except ValueError as e:
    #         print(f"Could not calculate AUC (possibly only one class present): {e}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    train_set = AgentErrorBench(file_paths="AgentErrorBench/train.json") 
    test_set = AgentErrorBench(file_paths="AgentErrorBench/test.json")
    
    print(f"Training on {len(train_set)} entries, Evaluating on {len(test_set)} entries...")
    
    # Increase Clusters to separate state space
    N_CLUSTERS = 30
    causal_model = CausalFeedbackModel(n_clusters=N_CLUSTERS) 
    vector_markov = VectorMarkovEntropy(device=device, n_clusters=N_CLUSTERS)
    
    print("\n=== Stage 1: Train Feature Projection (Triplet Loss) ===")
    causal_model.train_projection_layer(train_set)
    
    print("\n=== Stage 2: Quantize Action Space (Unified KMeans) ===")
    # Re-fit KMeans with larger K
    vector_markov.n_clusters = N_CLUSTERS
    vector_markov.fit_quantization(train_set, causal_model)
    
    print("\n=== Stage 3: Build Discrete Markov Model ===")
    vector_markov.build_markov_matrices(train_set, causal_model)
    
    # We now pass both kmeans models to training
    print("\n=== Stage 4: Train Proactive Prediction Head ===")
    causal_model.train_proactive_head(train_set, vector_markov) 
    
    print("\n=== Stage 4.5: Calibrate Risk Threshold ===")
    vector_markov.calibrate_threshold(train_set, causal_model)
    
    print("\n=== Stage 5: Evaluation (Proactive Risk) ===")
    run_causal_evaluation(causal_model, vector_markov, test_set)
