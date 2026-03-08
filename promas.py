import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
import glob
from collections import Counter, defaultdict
from tqdm import tqdm
import random

# Set Deterministic Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. Finite State Space Definitions ---

class AgentRole:
    DOER = "DOER"           # Writes Code, Files, Searches, Browses (High Impact)
    THINKER = "THINKER"     # Plans, Verifies, Manages, Chats (Mental/Safe)
    SYSTEM = "SYSTEM"       # Terminal, System Outputs (Passive)

    @classmethod
    def all(cls):
        return [cls.DOER, cls.THINKER, cls.SYSTEM]

class ActionType:
    ACT = "ACT"             # Write Code, Search, web-browse, write file
    TALK = "TALK"           # Plan, Chat, Verify, Thought
    OK = "OK"               # Execution Success
    FAIL = "FAIL"           # Execution Failure/Error
    
    @classmethod
    def all(cls):
        return [cls.ACT, cls.TALK, cls.OK, cls.FAIL]

class TaskType:
    DATA_ANALYSIS = "DATA_ANALYSIS" 
    INFO_RETRIEVAL = "INFO_RETRIEVAL" 
    LOGIC_PUZZLE = "LOGIC_PUZZLE" 
    GENERAL_ASSISTANCE = "GENERAL_ASSISTANCE"
    OTHER = "OTHER"
    
    @classmethod
    def all(cls):
        return [cls.DATA_ANALYSIS, cls.INFO_RETRIEVAL, cls.LOGIC_PUZZLE, cls.GENERAL_ASSISTANCE, cls.OTHER]

class LLMStateClassifier:
    """
    Uses a Local LLM to classify Agent Role and Action Type relative to the context.
    Replaces brittle if-else heuristics.
    """
    def __init__(self, model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        # Use auto device placement to avoid OOM on specific GPUs
        self.device_map = "auto"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve Model Path (Reuse logic)
        possible_paths = [
            "/home/ls/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots",
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        ]
        
        resolved_path = model_path
        for root_path in possible_paths:
            if os.path.exists(root_path) and os.path.isdir(root_path):
                subdirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
                if subdirs:
                    candidate = subdirs[0]
                    if os.path.exists(os.path.join(candidate, "config.json")):
                        resolved_path = candidate
                        break
        
        print(f"Loading LLM for State Classification from: {resolved_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            torch_dtype=torch.float16,
            device_map=self.device_map
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def classify_state(self, task_context, history_item, agent_description=""):
        content = history_item.get('content', '')[:1000]
        agent_name = history_item.get('name', history_item.get('role', 'Unknown'))
        
        # Coarse-grained System Prompt
        system_prompt = (
            "You are a specialized classifier for Multi-Agent Systems. "
            "Map the AGENT to one of [DOER, THINKER, SYSTEM] and ACTION to [ACT, TALK, OK, FAIL].\n"
            "\n"
            "GUIDELINES:\n"
            "1. DOER: Any agent that Writes Code, Edits Files, Browses Web, Searches DB, or performs concrete work.\n"
            "2. THINKER: Any agent that Plans, Verifies, Critiques, Manages, or Chats.\n"
            "3. SYSTEM: Computer Terminal, System Logs.\n"
            "\n"
            "1. ACT: Writing code/files, Searching, Clicking.\n"
            "2. TALK: Planning, Explanation, Verification, Thoughts.\n"
            "3. OK: Successful execution logs, Normal outputs.\n"
            "4. FAIL: Error messages, Exceptions, Stack traces, Failed commands.\n"
            "\n"
            "Output JSON format only: {\"role\": \"...\", \"action\": \"...\"}"
        )
        
        desc_text = f"Agent Description: {agent_description}\n" if agent_description else ""
        
        user_prompt = (
            f"Context: {task_context[:100]}\n"
            f"Agent: {agent_name}\n"
            f"{desc_text}"
            f"Content: {content}\n\n"
            "Classify."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)

        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=40,
                do_sample=False
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            json_str = generated_text.replace("```json", "").replace("```", "").strip()
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(json_str[start:end])
            else:
                data = {}
        except:
            data = {}
            
        role = data.get("role", AgentRole.THINKER) # Default safe
        action = data.get("action", ActionType.TALK)
        
        # Hard Constraints for Obvious Cases
        if agent_name == "Computer_terminal" or role == "SYSTEM":
            role = AgentRole.SYSTEM
            # Heuristic for fail vs ok if LLM missed it
            if "Error" in content or "Traceback" in content or "Exception" in content:
                action = ActionType.FAIL
            else:
                action = ActionType.OK
            
        if role not in AgentRole.all(): role = AgentRole.THINKER
        if action not in ActionType.all(): action = ActionType.TALK
        
        return TaskType.GENERAL_ASSISTANCE, role, action

class PreDefinedStateManager:
    """
    Wrapper that now uses LLM Classifier instead of if-else.
    Includes caching to avoid re-running LLM on same identical messages.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.cache = {}
    
    def extract_state(self, task_context, history_item, has_prior_error=False, agent_description=""):
        # Create a cache key from content hash + agent
        content_hash = hash(history_item.get('content', ''))
        key = (task_context[:50], history_item.get('name'), content_hash)
        
        if key in self.cache:
            task_type, role, action = self.cache[key]
        else:
            task_type, role, action = self.classifier.classify_state(task_context, history_item, agent_description=agent_description)
            self.cache[key] = (task_type, role, action)
            
        return (task_type, role, action, has_prior_error)

# --- 2. Neural Risk Model with Contrastive Learning ---

class NeuralRiskModel(nn.Module):
    """
    Uses Context and Auxiliary Task (Predicting Action) to learn risk scoring.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        self.task_types = {t: i for i, t in enumerate(TaskType.all())}
        self.roles = {r: i for i, r in enumerate(AgentRole.all())}
        self.actions = {a: i for i, a in enumerate(ActionType.all())}
        
        self.task_emb = nn.Embedding(len(self.task_types), 16)
        self.role_emb = nn.Embedding(len(self.roles), 16)
        self.action_emb = nn.Embedding(len(self.actions), 16)
        
        # prev encoded WITH action (Task, Role, PrevAction)
        prev_dim = 16 * 3  
        self.prev_encoder = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # curr encoded WITHOUT action (Task, Role)
        curr_dim = 16 * 2  
        self.curr_encoder = nn.Sequential(
            nn.Linear(curr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 1. Main Task: Predict Risk
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        
        # 2. Auxiliary Task: Predict Current Action
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.actions))
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
    
    def encode_prev_state(self, state_tuple):
        task_type, role, action = state_tuple
        
        task_idx = self.task_types.get(task_type, 0)
        role_idx = self.roles.get(role, 0)
        action_idx = self.actions.get(action, 0)
        
        task_t = torch.tensor([task_idx], dtype=torch.long, device=self.device)
        role_t = torch.tensor([role_idx], dtype=torch.long, device=self.device)
        action_t = torch.tensor([action_idx], dtype=torch.long, device=self.device)
        
        state_emb = torch.cat([self.task_emb(task_t), self.role_emb(role_t), self.action_emb(action_t)], dim=1)
        return state_emb

    def encode_curr_state(self, state_tuple):
        task_type, role = state_tuple
        
        task_idx = self.task_types.get(task_type, 0)
        role_idx = self.roles.get(role, 0)
        
        task_t = torch.tensor([task_idx], dtype=torch.long, device=self.device)
        role_t = torch.tensor([role_idx], dtype=torch.long, device=self.device)
        
        state_emb = torch.cat([self.task_emb(task_t), self.role_emb(role_t)], dim=1)
        return state_emb
    
    def forward(self, prev_state, curr_state):
        prev_emb = self.encode_prev_state(prev_state)  
        curr_emb = self.encode_curr_state(curr_state)  
        
        prev_encoded = self.prev_encoder(prev_emb)  
        curr_encoded = self.curr_encoder(curr_emb)  
        
        combined = torch.cat([prev_encoded, curr_encoded], dim=1)  
        
        risk = self.risk_head(combined) 
        action_logits = self.action_head(combined)
        
        return risk.squeeze().item(), action_logits
    
    def compute_loss(self, prev_states, curr_states, labels, target_actions):
        risks, action_preds = [], []
        for p_state, c_state in zip(prev_states, curr_states):
            prev_emb = self.encode_prev_state(p_state)
            curr_emb = self.encode_curr_state(c_state)
            
            combined = torch.cat([self.prev_encoder(prev_emb), self.curr_encoder(curr_emb)], dim=1)
            risks.append(self.risk_head(combined))
            action_preds.append(self.action_head(combined))
            
        risks = torch.cat(risks).squeeze(-1) 
        action_preds = torch.cat(action_preds, dim=0) 
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
        
        target_action_idx = [self.actions.get(a, 0) for a in target_actions]
        target_actions_t = torch.tensor(target_action_idx, dtype=torch.long, device=self.device)
        
        # Loss 1: BCE for Risk
        loss_risk = nn.BCELoss()(risks, labels_t)
        # Loss 2: CrossEntropy for Action Prediction (Auxiliary task)
        loss_action = nn.CrossEntropyLoss()(action_preds, target_actions_t)
        
        # Combined loss (weight context prediction by 0.5)
        return loss_risk + 0.5 * loss_action

class InitialRiskModel(nn.Module):
    """
    Learns risk for the very first step based on (TaskType, AgentRole).
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.task_types = {t: i for i, t in enumerate(TaskType.all())}
        self.roles = {r: i for i, r in enumerate(AgentRole.all())}
        
        self.task_emb = nn.Embedding(len(self.task_types), 16)
        self.role_emb = nn.Embedding(len(self.roles), 16)
        
        state_dim = 16 * 2
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        
    def encode_states(self, states):
        t_idx = [self.task_types.get(s[0], 0) for s in states]
        r_idx = [self.roles.get(s[1], 0) for s in states]
        
        t_t = torch.tensor(t_idx, dtype=torch.long, device=self.device)
        r_t = torch.tensor(r_idx, dtype=torch.long, device=self.device)
        
        return torch.cat([self.task_emb(t_t), self.role_emb(r_t)], dim=1)
        
    def forward(self, states):
        emb = self.encode_states(states)
        out = self.net(emb).squeeze(-1) # -> [batch_size]
        return out
        
    def compute_loss(self, states, labels):
        risks = self.forward(states)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
        return nn.BCELoss()(risks, labels_t)

# --- 2. Discrete Markov Chain ---

class DiscreteStateMarkov:
    """
    Hybrid Model:
    1. Hierarchical Markov Chain (for statistical priors)
    2. Neural Risk Model (for learned risk scoring via contrastive learning)
    3. Initial Risk Model (for 0th step specific evaluation)
    """
    def __init__(self, classifier):
        self.state_manager = PreDefinedStateManager(classifier)
        
        # Neural Risk Model
        self.neural_risk_model = NeuralRiskModel(hidden_dim=64)
        self.initial_risk_model = InitialRiskModel(hidden_dim=32)
        self.optimal_threshold = 0.1
        self.initial_optimal_threshold = 0.5
        
        # Training data for neural model
        self.training_transitions = []  # list of (prev_state, curr_state, is_mistake)
        self.training_initial_states = [] # list of (curr_state, is_mistake)

    def optimize_threshold(self, dataset):
        """"
        Find the best threshold that separates Safe steps from Mistake steps.
        """
        print("\nOptimizing Detection Threshold...")
        safe_scores = []
        mistake_scores = []

        initial_safe_scores = []
        initial_mistake_scores = []
        
        limit_files = len(dataset)
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            system_prompts = json_data.get('system_prompt', {})
            
            prev_state = None
            has_prior_error = False
            
            for i in range(len(history)):
                curr_msg = history[i]
                agent_name = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
                desc = system_prompts.get(agent_name, "")
                
                risk, state, _ = self.get_risk(
                    task_context, 
                    curr_msg, 
                    prev_state, 
                    has_prior_error, 
                    is_first_step=(i==0),
                    agent_description=desc
                )
                
                if i == 0:
                    if i == mistake_step:
                        initial_mistake_scores.append(risk)
                    else:
                        initial_safe_scores.append(risk)
                else:
                    if i == mistake_step:
                        mistake_scores.append(risk)
                    else:
                        safe_scores.append(risk)
                
                (_, _, action, _) = state
                if action == ActionType.FAIL: has_prior_error = True
                prev_state = state

        if mistake_scores:
            avg_safe = np.mean(safe_scores) if safe_scores else 0.0
            avg_mistake = np.mean(mistake_scores)
            std_safe = np.std(safe_scores) if safe_scores else 0.0
            
            from sklearn.cluster import KMeans
            all_scores = np.array(safe_scores + mistake_scores).reshape(-1, 1)
            if len(all_scores) >= 2:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(all_scores)
                centers = kmeans.cluster_centers_.flatten()
                self.optimal_threshold = float(np.mean(centers))
            else:
                self.optimal_threshold = 0.5
            
            print(f"Stats: Avg Safe Risk={avg_safe:.4f} (std={std_safe:.4f}), Avg Mistake Risk={avg_mistake:.4f}")
            print(f"Selected Optimal Threshold for transition (KMeans Boundary): {self.optimal_threshold:.4f}")

        if initial_mistake_scores:
            avg_safe_init = np.mean(initial_safe_scores) if initial_safe_scores else 0.0
            avg_mistake_init = np.mean(initial_mistake_scores)
            std_safe_init = np.std(initial_safe_scores) if initial_safe_scores else 0.0

            from sklearn.cluster import KMeans
            all_init_scores = np.array(initial_safe_scores + initial_mistake_scores).reshape(-1, 1)
            if len(all_init_scores) >= 2:
                kmeans_init = KMeans(n_clusters=2, random_state=42, n_init=10).fit(all_init_scores)
                centers_init = kmeans_init.cluster_centers_.flatten()
                self.initial_optimal_threshold = float(np.mean(centers_init))
            else:
                self.initial_optimal_threshold = 0.5
            
            print(f"Init Stats: Avg Safe Risk={avg_safe_init:.4f} (std={std_safe_init:.4f}), Avg Mistake Risk={avg_mistake_init:.4f}")
            print(f"Selected Optimal Threshold for initial step (KMeans Boundary): {self.initial_optimal_threshold:.4f}")
        else:
            if initial_safe_scores:
                self.initial_optimal_threshold = float(np.percentile(initial_safe_scores, 50))
                print(f"Init Stats: Avg Safe Risk={np.mean(initial_safe_scores):.4f}, using 50th percentile {self.initial_optimal_threshold:.4f}")

    def fit(self, dataset):
        print("Fitting Hybrid Markov Model with Neural Risk Learning...")
        limit_files = len(dataset)
        
        # Step 1: Collect training transitions
        print("\n[Step 1/3] Collecting training transitions...")
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            sys_prompts = json_data.get('system_prompt', {})
            
            prev_state_tuple = None
            has_prior_error = False
            
            for i in range(len(history)):
                curr_msg = history[i]
                agent_name = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
                desc = sys_prompts.get(agent_name, "")
                
                try:
                    curr_state = self.state_manager.extract_state(task_context, curr_msg, has_prior_error, desc)
                except: 
                    continue
                
                (c_t, c_r, c_a, c_e) = curr_state
                
                is_mistake = (i == mistake_step)
                
                if prev_state_tuple:
                    (p_t, p_r, p_a, p_e) = prev_state_tuple
                    
                    # Collect for neural model training
                    prev_neural_state = (p_t, p_r, p_a)   # Now includes prev_action!
                    curr_neural_state = (c_t, c_r)        # Curr action excluded to avoid direct leak
                    self.training_transitions.append((prev_neural_state, curr_neural_state, int(is_mistake), c_a)) # pass c_a as target
                elif i == 0:
                    curr_neural_state = (c_t, c_r)
                    self.training_initial_states.append((curr_neural_state, int(is_mistake)))
                
                if c_a == ActionType.FAIL: has_prior_error = True
                prev_state_tuple = curr_state
        
        # Step 2: Train neural risk model with contrastive learning
        print(f"\n[Step 2/3] Training Neural Risk Model on {len(self.training_transitions)} transitions...")
        self._train_neural_model()
        
        print(f"Training Initial Risk Model on {len(self.training_initial_states)} step-0 states...")
        self._train_initial_model()
        
        self.optimize_threshold(dataset)

    def _train_neural_model(self, epochs=20, batch_size=32, lr=0.001):
        """Train the neural risk model using contrastive learning"""
        optimizer = torch.optim.Adam(self.neural_risk_model.parameters(), lr=lr)
        
        # Prepare training data
        transitions = self.training_transitions
        n_batches = len(transitions) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle transitions
            indices = np.random.permutation(len(transitions))
            
            for batch_idx in range(0, len(transitions), batch_size):
                batch_indices = indices[batch_idx:batch_idx+batch_size]
                
                prev_states = [transitions[i][0] for i in batch_indices]
                curr_states = [transitions[i][1] for i in batch_indices]
                labels = [transitions[i][2] for i in batch_indices]
                target_actions = [transitions[i][3] for i in batch_indices]
                
                optimizer.zero_grad()
                loss = self.neural_risk_model.compute_loss(prev_states, curr_states, labels, target_actions)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        print(f"✓ Neural Risk Model trained successfully\n")
        self.neural_risk_model.eval()

    def _train_initial_model(self, epochs=10, batch_size=10, lr=0.01):
        """Train the initial risk model using contrastive learning"""
        optimizer = torch.optim.Adam(self.initial_risk_model.parameters(), lr=lr)
        states_data = self.training_initial_states
        n_batches = max(len(states_data) // batch_size, 1)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(len(states_data))
            
            for batch_idx in range(0, len(states_data), batch_size):
                batch_indices = indices[batch_idx:batch_idx+batch_size]
                states = [states_data[i][0] for i in batch_indices]
                labels = [states_data[i][1] for i in batch_indices]
                
                optimizer.zero_grad()
                loss = self.initial_risk_model.compute_loss(states, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            if (epoch + 1) % 5 == 0:
                print(f"  Initial Model Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        print(f"✓ Initial Risk Model trained successfully\n")
        self.initial_risk_model.eval()

    def get_risk(self, task_context, history_item, prev_state=None, has_prior_error=False, is_first_step=False, agent_description=""):
        state_full = self.state_manager.extract_state(task_context, history_item, has_prior_error, agent_description)
        (c_t, c_r, c_a, _) = state_full
        
        # If it's the first step, use the Initial Risk Model
        if is_first_step:
            try:
                with torch.no_grad():
                    curr_neural_state = (c_t, c_r)
                    initial_risk = self.initial_risk_model.forward([curr_neural_state])
                    # forward outputs a tensor of shape [batch_size], we extract .item()
                    risk_val = initial_risk[0].item()
                    return risk_val, state_full, "Neural_Initial"
            except:
                return 0.0, state_full, "Level0_FirstStep"
            
        # Primary: Use Neural Risk Model
        if prev_state and not is_first_step:
            (p_t, p_r, p_a, _) = prev_state
            prev_neural_state = (p_t, p_r, p_a)  # Include prev action
            curr_neural_state = (c_t, c_r)
            
            try:
                with torch.no_grad():
                    neural_risk, _ = self.neural_risk_model.forward(prev_neural_state, curr_neural_state)
                    return neural_risk, state_full, "Neural"
            except:
                pass  # Fall back to default if neural fails
        
        # Fallback: Return default safe risk
        return 0.0, state_full, "DefaultSafe"

# --- 3. Dataset & Evaluation ---

class WhoWhenDataset:
    def __init__(self, directory_path=None, file_paths=None):
        if file_paths:
            self.file_paths = file_paths
        elif directory_path:
            self.file_paths = glob.glob(os.path.join(directory_path, "*.json"))
        else:
            raise ValueError("Must provide either directory_path or file_paths")

    def load_file(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx): return self.load_file(self.file_paths[idx])

def run_evaluation(markov_model, dataset):
    print("\n--- Running Finite State Risk Evaluation (Neural Risk Model) ---")
    
    stats = {
        "total_cases": 0,
        "valid_cases": 0,
        "detected": 0,
        "early_detection": 0,
        "agent_hit": 0,
        "detection_with_correct_agent": 0
    }
    
    # Track which risk level was used for detection
    level_stats = {
        "Neural": {"attempted": 0, "detected": 0, "correct": 0},
        "Neural_Initial": {"attempted": 0, "detected": 0, "correct": 0},
        "Level0_FirstStep": {"attempted": 0, "detected": 0, "correct": 0},
        "DefaultSafe": {"attempted": 0, "detected": 0, "correct": 0},
        "No_Detection": {"attempted": 0, "detected": 0, "correct": 0}
    }
    
    RISK_THRESHOLD = markov_model.optimal_threshold
    INITIAL_RISK_THRESHOLD = getattr(markov_model, 'initial_optimal_threshold', RISK_THRESHOLD)
    print(f"Using Optimal Risk Threshold: {RISK_THRESHOLD:.4f}")
    if hasattr(markov_model, 'initial_optimal_threshold'):
        print(f"Using Initial Optimal Risk Threshold: {INITIAL_RISK_THRESHOLD:.4f}")
    
    detect_ratios = []
    
    for idx, json_data in enumerate(dataset):
        history = json_data.get('history', [])
        ms = json_data.get('mistake_step', -1)
        mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
        task_context = json_data.get('question', '')
        mistake_agent = json_data.get('mistake_agent', 'UNKNOWN')
        system_prompts = json_data.get('system_prompt', {})
        
        if mistake_step == -1: continue
        
        stats["total_cases"] += 1
        stats["valid_cases"] += 1
        
        detected_at = -1
        detected_agent = None
        detected_level = None
        outcome = "MISSED"
        agent_match = False
        
        limit = min(len(history), mistake_step + 5)
        
        prev_state = None
        has_prior_error = False
        max_observed_risk = 0.0
        
        for i in range(limit):
            curr_msg = history[i]
            agent_name = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
            agent_description = system_prompts.get(agent_name, "")
            
            risk, state, used_level = markov_model.get_risk(
                task_context, 
                curr_msg, 
                prev_state=prev_state, 
                has_prior_error=has_prior_error, 
                is_first_step=(i==0),
                agent_description=agent_description
            )

            # Record attempt
            if used_level in level_stats:
                level_stats[used_level]["attempted"] += 1
            
            if i == mistake_step:
                max_observed_risk = risk
            
            threshold = INITIAL_RISK_THRESHOLD if i == 0 else RISK_THRESHOLD
            
            # Only detect if not yet detected
            if detected_at == -1 and risk > threshold:
                detected_at = i
                detected_agent = agent_name
                detected_level = used_level
            
            (_, _, action, _) = state
            if action == ActionType.FAIL:
                has_prior_error = True
            
            prev_state = state
        
        if detected_at != -1:
            agent_match = detected_agent == mistake_agent
            if agent_match:
                stats["agent_hit"] += 1
            
            # Determine detection type and correctness
            if detected_at == mistake_step:
                stats["detected"] += 1
                if agent_match:
                    stats["detection_with_correct_agent"] += 1
                outcome = "EXACT_HIT"
                is_correct = True
            elif detected_at < mistake_step:
                stats["early_detection"] += 1
                if agent_match:
                    stats["detection_with_correct_agent"] += 1
                outcome = "EARLY_WARN"
                is_correct = False
            else:
                outcome = "LATE_MATCH"
                is_correct = False
            
            # Record detection statistics
            if detected_level in level_stats:
                level_stats[detected_level]["detected"] += 1
                if is_correct:
                    level_stats[detected_level]["correct"] += 1
            
            detect_ratios.append(detected_at / len(history))
            agent_status = "✓" if agent_match else "✗"
            print(f"[ID {idx}] {outcome} @ {detected_at} (Tgt: {mistake_step}) Agent: {detected_agent} {agent_status} (Expected: {mistake_agent}) [{detected_level}]")
        else:
            level_stats["No_Detection"]["attempted"] += 1
            detect_ratios.append(1.0)
            print(f"[ID {idx}] MISSED {mistake_step} (Risk @ Mistake Step: {max_observed_risk:.4f})")

    print("\n=== Final Results ===")
    print(f"Total Valid Cases: {stats['valid_cases']}")
    print(f"Exact Hit: {stats['detected']} ({stats['detected']/stats['valid_cases']:.2%})")
    print(f"Early Warning: {stats['early_detection']} ({stats['early_detection']/stats['valid_cases']:.2%})")
    combined = stats['detected'] + stats['early_detection']
    print(f"Combined Recall (Exact+Early): {combined/stats['valid_cases']:.2%}")
    print(f"\nAgent Hit Rate: {stats['agent_hit']} ({stats['agent_hit']/stats['valid_cases']:.2%})")
    print(f"Detection with Correct Agent: {stats['detection_with_correct_agent']} ({stats['detection_with_correct_agent']/stats['valid_cases']:.2%})")
    print(f"Avg Detection position ratio: {np.mean(detect_ratios):.4f}")
    
    # Risk Level Statistics
    print(f"\n=== Risk Level Usage & Hit Rate Statistics ===")
    for level_name in ["Neural", "Neural_Initial", "Level0_FirstStep", "DefaultSafe"]:
        if level_name in level_stats:
            level_data = level_stats[level_name]
            attempted = level_data["attempted"]
            detected = level_data["detected"]
            correct = level_data["correct"]
            
            if attempted > 0:
                detection_rate = detected / attempted
                print(f"\n{level_name}:")
                print(f"  Attempted: {attempted}")
                print(f"  Detected: {detected} ({detection_rate:.2%})")
                if detected > 0:
                    hit_rate = correct / detected
                    print(f"  Hit Rate: {correct}/{detected} ({hit_rate:.2%})")
                else:
                    print(f"  Hit Rate: 0/0 (N/A)")
    
    # No Detection statistics
    no_detection_count = level_stats["No_Detection"]["attempted"]
    print(f"\n\nNo Detection (Missed):")
    print(f"  Cases: {no_detection_count} ({no_detection_count/stats['valid_cases']:.2%})")


if __name__ == "__main__":
    # data_dir = "Who&When/Algorithm-Generated"
    data_dir = "Who&When/Hand-Crafted"
    all_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    # Shuffle
    random.shuffle(all_files)
    
    split = int(len(all_files) * 0.4)
    train_files = all_files[:split]
    test_files = all_files[split:]
    
    train_set = WhoWhenDataset(file_paths=train_files)
    test_set = WhoWhenDataset(file_paths=test_files)
    
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    # Initialize LLM Classifier
    classifier = LLMStateClassifier()
    
    markov = DiscreteStateMarkov(classifier)
    markov.fit(train_set)
    
    # Evaluate on Test
    run_evaluation(markov, test_set)
