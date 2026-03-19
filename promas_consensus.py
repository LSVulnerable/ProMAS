import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import glob
import re
from collections import Counter, defaultdict
from tqdm import tqdm
import random
import math

# Set Deterministic Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. Finite State Space Definitions ---
class AgentRole:
    DOER = "DOER"                 # Concrete work (Code, file, browse)
    THINKER = "THINKER"           # Deep thinking, verify, critique
    MANAGER = "MANAGER"           # Orchestrator, dispatching tasks
    USER_PROXY = "USER_PROXY"     # End user or proxy standing in for user
    SYSTEM = "SYSTEM"             # Terminal, tool outputs

    @classmethod
    def all(cls):
        return [cls.DOER, cls.THINKER, cls.MANAGER, cls.USER_PROXY, cls.SYSTEM]

class ActionType:
    ACT = "ACT"                   # Execute code, read/write files
    TOOL_CALL = "TOOL_CALL"       # Decide to invoke a tool/agent
    PLANNING = "PLANNING"         # Step-by-step thinking
    EVALUATION = "EVALUATION"     # Reviewing other's responses
    INFORMATION = "INFORMATION"   # Providing raw prompt/info
    OK = "OK"                     # Success indicator
    FAIL = "FAIL"                 # Error, generic failure
    
    @classmethod
    def all(cls):
        return [cls.ACT, cls.TOOL_CALL, cls.PLANNING, cls.EVALUATION, cls.INFORMATION, cls.OK, cls.FAIL]

class TaskType:
    DATA_ANALYSIS = "DATA_ANALYSIS" 
    INFO_RETRIEVAL = "INFO_RETRIEVAL" 
    LOGIC_PUZZLE = "LOGIC_PUZZLE" 
    GENERAL_ASSISTANCE = "GENERAL_ASSISTANCE"
    OTHER = "OTHER"
    
    @classmethod
    def all(cls):
        return [cls.DATA_ANALYSIS, cls.INFO_RETRIEVAL, cls.LOGIC_PUZZLE, cls.GENERAL_ASSISTANCE, cls.OTHER]
class ConsensusState:
    NOVEL_PROPOSAL = "NOVEL_PROPOSAL"
    GROUNDED_CONSENSUS = "GROUNDED_CONSENSUS"
    BLIND_CONSENSUS = "BLIND_CONSENSUS"
    CRITICAL_DIVERGENCE = "CRITICAL_DIVERGENCE"
    STUCK_LOOP = "STUCK_LOOP"

    @classmethod
    def all(cls):
        return [cls.NOVEL_PROPOSAL, cls.GROUNDED_CONSENSUS, cls.BLIND_CONSENSUS, cls.CRITICAL_DIVERGENCE, cls.STUCK_LOOP]

class LLMStateClassifier:
    """
    Uses a local LLM to classify predefined agent/action labels and
    a compact trajectory state (progress + consensus).
    """
    def __init__(self, model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.device_map = "auto"
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

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

    def classify_state(self, task_context, history_item, agent_description="", history_context=""):
        if isinstance(task_context, dict):
            task_context_str = task_context.get("prompt", str(task_context))
        else:
            task_context_str = str(task_context)

        content = str(history_item.get("content", "") or "")[:1200]
        agent_name = history_item.get("name", history_item.get("role", "Unknown"))
        context = f"Prior context: {history_context[:800]}\n" if history_context else ""
        desc = f"Agent description: {str(agent_description)[:300]}\n" if agent_description else ""

        system_prompt = (
            "You classify one multi-agent turn with fixed labels.\n"
            "Return only valid JSON.\n"
            "Fields:\n"
            "1) agent_role: DOER|THINKER|MANAGER|USER_PROXY|SYSTEM\n"
            "2) action_type: ACT|TOOL_CALL|PLANNING|EVALUATION|INFORMATION|OK|FAIL\n"
            "3) consensus_state: NOVEL_PROPOSAL|GROUNDED_CONSENSUS|BLIND_CONSENSUS|CRITICAL_DIVERGENCE|STUCK_LOOP\n"
            "JSON schema:\n"
            "{\"agent_role\":\"...\",\"action_type\":\"...\",\"consensus_state\":\"...\"}"
        )

        user_prompt = (
            f"Task Context: {task_context_str[:220]}\n"
            f"Speaker: {agent_name}\n"
            f"{desc}"
            f"{context}"
            f"Current Message: {content}\n"
            "Classify now."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=40,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        try:
            json_str = generated_text.replace("```json", "").replace("```", "").strip()
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            data = json.loads(json_str[start:end]) if (start != -1 and end != -1) else {}
        except Exception:
            data = {}

        role = data.get("agent_role", AgentRole.DOER)
        action = data.get("action_type", ActionType.INFORMATION)
        consensus = data.get("consensus_state", ConsensusState.NOVEL_PROPOSAL)

        if role not in AgentRole.all():
            role = AgentRole.DOER
        if action not in ActionType.all():
            action = ActionType.INFORMATION
        if consensus not in ConsensusState.all():
            consensus = ConsensusState.NOVEL_PROPOSAL

        return role, action, consensus

    def classify_task_type(self, task_context):
        if isinstance(task_context, dict):
            task_context_str = task_context.get("prompt", str(task_context))
        else:
            task_context_str = str(task_context)

        system_prompt = (
            "Classify the user task into one fixed label. Return only valid JSON.\n"
            "task_type: DATA_ANALYSIS|INFO_RETRIEVAL|LOGIC_PUZZLE|GENERAL_ASSISTANCE|OTHER\n"
            "JSON schema: {\"task_type\":\"...\"}"
        )
        user_prompt = f"Task: {task_context_str[:800]}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=20,
                do_sample=False,
            )
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        try:
            json_str = generated_text.replace("```json", "").replace("```", "").strip()
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            data = json.loads(json_str[start:end]) if (start != -1 and end != -1) else {}
        except Exception:
            data = {}

        task_type = data.get("task_type", TaskType.OTHER)
        if task_type not in TaskType.all():
            task_type = TaskType.OTHER
        return task_type

class PreDefinedStateManager:
    """
    Wrapper that now uses LLM Classifier instead of if-else.
    Includes caching to avoid re-running LLM on same identical messages.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.cache = {}
        self.task_cache = {}
    
    def extract_state(self, task_context, history_item, agent_description="", history_context=""):
        if isinstance(task_context, dict):
            task_context_str = task_context.get('prompt', str(task_context))
        else:
            task_context_str = str(task_context)
        
        content_hash = hash(history_item.get('content', ''))
        key = (task_context_str[:50], content_hash, hash(history_context[:240]))
        
        if key in self.cache:
            role, action, consensus = self.cache[key]
        else:
            role, action, consensus = self.classifier.classify_state(
                task_context,
                history_item,
                agent_description=agent_description,
                history_context=history_context
            )
            self.cache[key] = (role, action, consensus)
            
        content_text = history_item.get('content', '') or ''
        return (role, action, consensus, content_text)

    def extract_task_state(self, task_context):
        if isinstance(task_context, dict):
            task_context_str = task_context.get('prompt', str(task_context))
        else:
            task_context_str = str(task_context)

        key = hash(task_context_str[:800])
        if key in self.task_cache:
            return self.task_cache[key]

        try:
            task_type = self.classifier.classify_task_type(task_context)
        except Exception:
            task_type = TaskType.OTHER

        consensus_map = {
            TaskType.DATA_ANALYSIS: ConsensusState.GROUNDED_CONSENSUS,
            TaskType.INFO_RETRIEVAL: ConsensusState.NOVEL_PROPOSAL,
            TaskType.LOGIC_PUZZLE: ConsensusState.CRITICAL_DIVERGENCE,
            TaskType.GENERAL_ASSISTANCE: ConsensusState.NOVEL_PROPOSAL,
            TaskType.OTHER: ConsensusState.NOVEL_PROPOSAL,
        }
        state = (
            AgentRole.USER_PROXY,
            ActionType.INFORMATION,
            consensus_map.get(task_type, ConsensusState.NOVEL_PROPOSAL),
            task_context_str,
        )
        self.task_cache[key] = state
        return state

class TextFeatureExtractor:
    """
    Encodes message content into dense semantic vectors for risk modeling.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.cache = {}

    def encode(self, text):
        if text is None:
            text = ""
        text = str(text).strip()
        if text in self.cache:
            return self.cache[text]
        vec = self.encoder.encode(text, normalize_embeddings=True)
        self.cache[text] = vec
        return vec

# --- 2. Neural Risk Model with Contrastive Learning ---

class NeuralRiskModel(nn.Module):
    """
    Risk scorer over Markov node state: (prev_action, curr_consensus, next_role).
    """
    def __init__(self, hidden_dim=128, text_dim=384, dropout=0.2, focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()

        self.actions = {c: i for i, c in enumerate(ActionType.all())}
        self.consensuses = {c: i for i, c in enumerate(ConsensusState.all())}
        self.roles = {c: i for i, c in enumerate(AgentRole.all())}

        emb_dim = 8
        self.prev_action_emb = nn.Embedding(len(self.actions), emb_dim)
        self.consensus_emb = nn.Embedding(len(self.consensuses), emb_dim)
        self.next_role_emb = nn.Embedding(len(self.roles), emb_dim)

        self.text_dim = text_dim
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        state_dim = (emb_dim * 3) + self.text_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def _to_text_tensor(self, text_vector):
        vec = np.asarray(text_vector, dtype=np.float32).flatten()
        if vec.shape[0] != self.text_dim:
            fixed = np.zeros((self.text_dim,), dtype=np.float32)
            copy_len = min(self.text_dim, vec.shape[0])
            fixed[:copy_len] = vec[:copy_len]
            vec = fixed
        return torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _logic_features(self, task_text, prev_text, curr_text):
        task_t = self._to_text_tensor(task_text)
        prev_t = self._to_text_tensor(prev_text)
        curr_t = self._to_text_tensor(curr_text)

        cos_sim = F.cosine_similarity(prev_t, curr_t, dim=1, eps=1e-8).unsqueeze(1)
        cos_task_prev = F.cosine_similarity(task_t, prev_t, dim=1, eps=1e-8).unsqueeze(1)
        cos_task_curr = F.cosine_similarity(task_t, curr_t, dim=1, eps=1e-8).unsqueeze(1)

        semantic_delta = torch.mean(torch.abs(curr_t - prev_t), dim=1, keepdim=True)
        task_prev_delta = torch.mean(torch.abs(task_t - prev_t), dim=1, keepdim=True)
        task_curr_delta = torch.mean(torch.abs(task_t - curr_t), dim=1, keepdim=True)

        feats = torch.cat(
            [
                cos_sim,
                cos_task_prev,
                cos_task_curr,
                semantic_delta,
                task_prev_delta,
                task_curr_delta,
            ],
            dim=1,
        )
        return feats

    def encode_node_state(self, node_state, curr_text_vector):
        prev_action, curr_consensus, next_role = node_state
        action_t = torch.tensor([self.actions.get(prev_action, 0)], dtype=torch.long, device=self.device)
        consensus_t = torch.tensor([self.consensuses.get(curr_consensus, 0)], dtype=torch.long, device=self.device)
        role_t = torch.tensor([self.roles.get(next_role, 0)], dtype=torch.long, device=self.device)
        text_t = self._to_text_tensor(curr_text_vector)

        return torch.cat(
            [
                self.prev_action_emb(action_t),
                self.consensus_emb(consensus_t),
                self.next_role_emb(role_t),
                text_t,
            ],
            dim=1,
        )

    def forward(self, node_state, task_text, prev_text, curr_text):
        node_emb = self.encode_node_state(node_state, curr_text)
        node_encoded = self.node_encoder(node_emb)
        logic_feats = self._logic_features(task_text, prev_text, curr_text)
        combined = torch.cat([node_encoded, logic_feats], dim=1)
        risk = self.risk_head(combined)
        return risk.squeeze().item()

    def focal_bce_loss(self, probs, labels, sample_weights=None):
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(probs, labels, reduction='none')
        pt = torch.where(labels == 1, probs, 1 - probs)
        alpha_t = torch.where(labels == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal = alpha_t * ((1 - pt) ** self.focal_gamma) * bce
        if sample_weights is not None:
            focal = focal * sample_weights
        return focal.mean()

    def compute_loss(self, node_states, task_texts, prev_texts, curr_texts, labels, sample_weights=None):
        risks = []
        for n_state, t_txt, p_txt, c_txt in zip(node_states, task_texts, prev_texts, curr_texts):
            node_emb = self.encode_node_state(n_state, c_txt)
            node_encoded = self.node_encoder(node_emb)
            logic_feats = self._logic_features(t_txt, p_txt, c_txt)
            combined = torch.cat([node_encoded, logic_feats], dim=1)
            risks.append(self.risk_head(combined))

        risks = torch.cat(risks).squeeze(-1)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
        
        sample_weights_t = None
        if sample_weights is not None:
            sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=self.device)
            
        return self.focal_bce_loss(risks, labels_t, sample_weights=sample_weights_t)


class InitialRiskModel(nn.Module):
    """
    Learns risk for the very first step based on (TaskType, initial_role).
    """
    def __init__(self, hidden_dim=64, text_dim=384, dropout=0.2):
        super().__init__()
        self.task_types = {t: i for i, t in enumerate(TaskType.all())}
        self.roles = {r: i for i, r in enumerate(AgentRole.all())}
        
        self.task_emb = nn.Embedding(len(self.task_types), 16)
        self.role_emb = nn.Embedding(len(self.roles), 16)
        self.text_dim = text_dim
        
        state_dim = (16 * 2) + self.text_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        
    def encode_states(self, states, text_vectors):
        t_idx = [self.task_types.get(s[0], 0) for s in states]
        r_idx = [self.roles.get(s[1], 0) for s in states]
        
        t_t = torch.tensor(t_idx, dtype=torch.long, device=self.device)
        r_t = torch.tensor(r_idx, dtype=torch.long, device=self.device)
        
        txt_arr = np.asarray(text_vectors, dtype=np.float32).flatten()
        if len(states) == 1:
            txt_arr = txt_arr.reshape(1, -1)
        else:
            txt_arr = txt_arr.reshape(len(states), -1)
            
        if txt_arr.shape[1] != self.text_dim:
            fixed = np.zeros((txt_arr.shape[0], self.text_dim), dtype=np.float32)
            copy_len = min(self.text_dim, txt_arr.shape[1])
            fixed[:, :copy_len] = txt_arr[:, :copy_len]
            txt_arr = fixed
        txt_t = torch.tensor(txt_arr, dtype=torch.float32, device=self.device)

        return torch.cat([self.task_emb(t_t), self.role_emb(r_t), txt_t], dim=1)
        
    def forward(self, states, text_vectors):
        emb = self.encode_states(states, text_vectors)
        out = self.net(emb).squeeze(-1)
        return out
        
    def compute_loss(self, states, text_vectors, labels):
        risks = self.forward(states, text_vectors)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
        if risks.dim() == 0:
            risks = risks.unsqueeze(0)
            labels_t = labels_t.unsqueeze(0)
        return nn.BCELoss()(risks, labels_t)

class DiscreteStateMarkov:

    """
    Hybrid Model:
    1. Hierarchical Markov Chain (for statistical priors)
    2. Neural Risk Model (for learned risk scoring via contrastive learning)
    3. Node state is fixed as (prev_action, curr_consensus, next_role).
    """
    def __init__(self, classifier):
        self.state_manager = PreDefinedStateManager(classifier)
        self.text_extractor = TextFeatureExtractor()

        self.neural_risk_model = NeuralRiskModel(hidden_dim=128, text_dim=self.text_extractor.dim)
        self.initial_risk_model = InitialRiskModel(hidden_dim=64, text_dim=self.text_extractor.dim)
        self.initial_optimal_threshold = 0.5
        self.training_initial_states = []
        self.optimal_threshold = 0.5
        self.transition_thresholds = {}
        self.transition_counts = defaultdict(int)
        self.transition_failure_sums = defaultdict(float)
        self.global_impact_sum = 0.0
        self.global_impact_count = 0
        self.global_hazard_rate = 0.5
        self.training_transitions = []
        self.neighbor_positive_weight = 0.25
        self.class_pos_weight = 1.0

    def _node_state(self, prev_state, next_state):
        _, prev_action, curr_consensus, _ = prev_state
        next_role, _, _, _ = next_state
        return (prev_action, curr_consensus, next_role)

    def _task_context_to_text(self, task_context):
        if isinstance(task_context, dict):
            return str(task_context.get('prompt', str(task_context)))
        return str(task_context)

    def _estimate_transition_hazard(self, transition_key):
        total = self.transition_counts.get(transition_key, 0)
        hazard = self.transition_failure_sums.get(transition_key, 0.0)
        if total == 0:
            return float(self.global_hazard_rate)
        return float(hazard / total)

    def _transition_label(self, step_idx, mistake_step):
        effective_step = step_idx + 1
        if mistake_step < 0:
            return 0.0
        if effective_step == mistake_step:
            return 1.0
        # Soft positives around the mistake step reduce extreme one-positive sparsity.
        if abs(effective_step - mistake_step) == 1:
            return float(self.neighbor_positive_weight)
        return 0.0

    def _initial_label(self, mistake_step):
        if mistake_step < 0:
            return 0.0
        if mistake_step == 0:
            return 1.0
        if mistake_step == 1:
            return float(self.neighbor_positive_weight)
        return 0.0

    def _build_history_context(self, history, step_idx, window=8):
        if not history or step_idx <= 0:
            return ""
        start = max(0, step_idx - window)
        snippets = []
        for item in history[start:step_idx]:
            agent = item.get('name', item.get('role', 'Unknown'))
            content = str(item.get('content', '') or '').strip().replace("\n", " ")
            if len(content) > 200:
                content = content[:200] + "..."
            snippets.append(f"{agent}: {content}")
        return " | ".join(snippets)

    def get_transition_threshold(self, step_idx, total_steps, transition_key=None):
        if transition_key == "InitialState" or step_idx == 0:
            return float(self.initial_optimal_threshold)
        if transition_key is not None and transition_key in self.transition_thresholds:
            return float(self.transition_thresholds[transition_key])
        return float(self.optimal_threshold)

    def normalize_agent_name(self, agent_name):
        if agent_name is None:
            return "unknown"
        name = str(agent_name).strip()
        if "->" in name:
            rhs = name.split("->", 1)[1].strip().replace(")", "").strip()
            if rhs:
                name = rhs
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        return name.lower()

    def canonical_agent_label(self, agent_name):
        name = self.normalize_agent_name(agent_name)
        name = re.sub(r"[^a-z0-9\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        name = re.sub(r"\b([a-z]+)\d+[a-z0-9]*\b$", r"\1", name).strip()
        name = re.sub(r"\b\d+[a-z0-9]*\b$", "", name).strip()
        return name

    def resolve_agent_description(self, system_prompts, agent_name):
        if not system_prompts:
            return ""

        if agent_name in system_prompts:
            return system_prompts[agent_name]

        norm = self.normalize_agent_name(agent_name)
        for k, v in system_prompts.items():
            if self.normalize_agent_name(k) == norm:
                return v

        canonical = self.canonical_agent_label(agent_name)
        for k, v in system_prompts.items():
            if self.canonical_agent_label(k) == canonical:
                return v

        return ""

    def calibrate_risk(self, risk, agent_name):
        return float(np.clip(float(risk), 0.0, 1.0))

    def _simulate_detection(self, true_labels, risks):
        if len(risks) == 0:
            return 0.0

        risks = np.array(risks)
        true_labels = np.array(true_labels)

        pred_labels = (risks >= self.optimal_threshold).astype(int)

        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def optimize_threshold(self, dataset):
        from sklearn.cluster import KMeans
        import numpy as np
        
        initial_probs = []
        initial_labels = []
        neural_probs = []
        neural_labels = []

        self.neural_risk_model.eval()
        self.initial_risk_model.eval()
        
        limit_files = len(dataset)
        for idx in range(limit_files):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            total_steps = len(history)
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            task_type = json_data.get('task_target', '')
            system_prompts = json_data.get('system_prompt', {})

            if total_steps > 0:
                first_role = self.normalize_agent_name(history[0].get('name', history[0].get('role', 'Unknown')))
                if task_type not in TaskType.all():
                    try:
                        task_type = self.state_manager.classifier.classify_task_type(task_context)
                    except Exception:
                        task_type = TaskType.OTHER
                lbl = self._initial_label(mistake_step)
                text_info = str(history[0].get('content', ''))
                initial_vec = self.text_extractor.encode(text_info)
                p_err, _, _ = self.get_transition_risk(None, None, -1, is_initial=True, 
                                                       initial_state=(task_type, first_role), initial_text=initial_vec)
                initial_probs.append(p_err)
                initial_labels.append(1 if lbl >= 0.5 else 0)

            history_states = []
            for i in range(total_steps):
                curr_msg = history[i]
                next_msg = history[i + 1] if i + 1 < len(history) else None
                if next_msg is None:
                    break
                    
                curr_agent_name = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
                next_agent_name = next_msg.get('name', next_msg.get('role', 'Unknown'))
                
                curr_desc = self.resolve_agent_description(system_prompts, curr_agent_name)
                next_desc = self.resolve_agent_description(system_prompts, next_agent_name)
                
                try:
                    curr_history_ctx = self._build_history_context(history, i)
                    next_history_ctx = self._build_history_context(history, i + 1)
                    curr_state_tuple = self.state_manager.extract_state(
                        task_context, curr_msg, curr_desc, history_context=curr_history_ctx
                    )
                    next_state_tuple = self.state_manager.extract_state(
                        task_context, next_msg, next_desc, history_context=next_history_ctx
                    )
                except Exception:
                    continue
                    
                (p_role, p_action, p_consensus, p_txt) = curr_state_tuple
                (c_role, c_action, c_consensus, c_txt) = next_state_tuple
                node_state = (p_action, p_consensus, c_role)

                task_context_text = self._task_context_to_text(task_context)
                task_text_vec = self.text_extractor.encode(task_context_text)
                prev_text_vec = self.text_extractor.encode(p_txt)
                curr_text_vec = self.text_extractor.encode(c_txt)
                
                p_err, _, _ = self.get_transition_risk(task_context, history, i, transition_state=((node_state, (task_type, c_role), prev_text_vec, curr_text_vec)))
                lbl = self._transition_label(i + 1, mistake_step)
                
                neural_probs.append(p_err)
                neural_labels.append(1 if lbl >= 0.5 else 0)

        def get_optimal_thresh(probs, labels):
            if not probs: return 0.5
            pos_probs = [p for p, l in zip(probs, labels) if l == 1]
            neg_probs = [p for p, l in zip(probs, labels) if l == 0]
            
            if not pos_probs or not neg_probs:
                return np.mean(probs)
                
            X = np.array(probs).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
            centers = sorted(kmeans.cluster_centers_.flatten())
            return float((centers[0] + centers[1]) / 2.0)
            
        self.initial_optimal_threshold = get_optimal_thresh(initial_probs, initial_labels)
        self.optimal_threshold = get_optimal_thresh(neural_probs, neural_labels)
        
        print(f"Optimized Initial Threshold via KMeans: {self.initial_optimal_threshold:.4f}")
        print(f"Optimized Neural Threshold via KMeans: {self.optimal_threshold:.4f}")

        return self.optimal_threshold

    def fit(self, dataset):
        print("Fitting Markov + Neural model with compact state...")
        limit_files = len(dataset)
        self.training_transitions = []
        self.training_initial_states = []
        self.transition_counts = defaultdict(int)
        self.transition_failure_sums = defaultdict(float)
        self.global_impact_sum = 0.0
        self.global_impact_count = 0

        print("\n[Step 1/3] Collecting training transitions...")
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            total_steps = len(history)
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            task_context_text = self._task_context_to_text(task_context)
            task_text_vec = self.text_extractor.encode(task_context_text)
            sys_prompts = json_data.get('system_prompt', {})

            for i in range(-1, len(history) - 1):
                if i == -1 and not history:
                    continue

                if i == -1:
                    next_msg = history[0]
                    next_agent_name = self.normalize_agent_name(next_msg.get('name', next_msg.get('role', 'Unknown')))
                    text_info = str(next_msg.get('content', ''))
                    initial_vec = self.text_extractor.encode(text_info)
                    label = self._initial_label(mistake_step)

                    task_type = json_data.get('task_target', TaskType.OTHER)
                    if task_type not in TaskType.all():
                        try:
                            task_type = self.state_manager.classifier.classify_task_type(task_context)
                        except Exception:
                            task_type = TaskType.OTHER
                        
                    self.training_initial_states.append(
                        ((task_type, next_agent_name), initial_vec, 1.0 if label >= 0.5 else 0.0)
                    )
                    continue
                else:
                    curr_msg = history[i]
                    next_msg = history[i + 1]
                    curr_agent_name = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
                    next_agent_name = next_msg.get('name', next_msg.get('role', 'Unknown'))
                    curr_desc = self.resolve_agent_description(sys_prompts, curr_agent_name)
                    next_desc = self.resolve_agent_description(sys_prompts, next_agent_name)

                    try:
                        curr_history_ctx = self._build_history_context(history, i)
                        next_history_ctx = self._build_history_context(history, i + 1)
                        curr_state = self.state_manager.extract_state(
                            task_context,
                            curr_msg,
                            curr_desc,
                            history_context=curr_history_ctx,
                        )
                        next_state = self.state_manager.extract_state(
                            task_context,
                            next_msg,
                            next_desc,
                            history_context=next_history_ctx,
                        )
                    except Exception:
                        continue
                (p_role, p_action, p_consensus, p_txt) = curr_state
                (c_role, c_action, c_consensus, c_txt) = next_state

                node_state = (p_action, p_consensus, c_role)

                prev_text_vec = self.text_extractor.encode(p_txt)
                curr_text_vec = self.text_extractor.encode(c_txt)
                transition_label = self._transition_label(i, mistake_step)

                t_key = node_state
                self.transition_counts[t_key] += 1
                self.transition_failure_sums[t_key] += float(transition_label)

                self.global_impact_sum += float(transition_label)
                self.global_impact_count += 1

                self.training_transitions.append(
                    (
                        node_state,
                        task_text_vec,
                        prev_text_vec,
                        curr_text_vec,
                        float(transition_label),
                        c_action,
                        1.0,
                    )
                )

        if self.transition_counts:
            total_hazard = float(sum(self.transition_failure_sums.values()))
            total_transitions = float(sum(self.transition_counts.values()))
            self.global_hazard_rate = (total_hazard / total_transitions) if total_transitions > 0 else 0.5
        else:
            self.global_hazard_rate = 0.5

        hard_pos = sum(1 for t in self.training_transitions if t[4] >= 0.5)
        hard_neg = max(1, len(self.training_transitions) - hard_pos)
        if hard_pos > 0:
            self.class_pos_weight = float(np.clip(hard_neg / hard_pos, 1.0, 20.0))
        else:
            self.class_pos_weight = 1.0
        print(
            f"Class balance: hard_pos={hard_pos}, hard_neg={hard_neg}, "
            f"pos_weight={self.class_pos_weight:.2f}"
        )
        
        print(f"\n[Step 2/3] Training Models...")
        print(f"Training Initial Risk Model on {len(self.training_initial_states)} initial states...")
        self._train_initial_model()
        print(f"Training Neural Risk Model on {len(self.training_transitions)} transitions...")
        self._train_neural_model()
        
        self.optimize_threshold(dataset)

    def _train_initial_model(self, epochs=20, batch_size=32, lr=0.001):
        optimizer = torch.optim.Adam(self.initial_risk_model.parameters(), lr=lr, weight_decay=1e-4)
        best_loss = float('inf')
        patience_counter = 0

        if not self.training_initial_states:
            print("No initial states to train.")
            return

        for epoch in range(epochs):
            self.initial_risk_model.train()
            epoch_loss = 0.0
            np.random.shuffle(self.training_initial_states)

            for i in range(0, len(self.training_initial_states), batch_size):
                batch = self.training_initial_states[i:i+batch_size]

                states = [b[0] for b in batch]
                text_vecs = [b[1] for b in batch]
                labels = [b[2] for b in batch]

                optimizer.zero_grad()
                loss = self.initial_risk_model.compute_loss(states, text_vecs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(self.training_initial_states) // batch_size)
            if (epoch + 1) % 5 == 0:
                print(f"  Initial Model Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")

        print(f"✓ Initial Risk Model trained successfully")
        self.initial_risk_model.eval()

    def _train_neural_model(self, epochs=20, batch_size=32, lr=0.001):
        optimizer = torch.optim.Adam(self.neural_risk_model.parameters(), lr=lr, weight_decay=1e-4)

        if not self.training_transitions:
            print("No transitions to train.")
            return

        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(self.training_transitions)

            for i in range(0, len(self.training_transitions), batch_size):
                batch = self.training_transitions[i:i+batch_size]

                node_states = [t[0] for t in batch]
                task_texts = [t[1] for t in batch]
                prev_texts = [t[2] for t in batch]
                curr_texts = [t[3] for t in batch]
                labels = [t[4] for t in batch]
                sample_weights = [t[6] for t in batch]

                optimizer.zero_grad()
                loss = self.neural_risk_model.compute_loss(
                    node_states, task_texts, prev_texts, curr_texts,
                    labels, sample_weights=sample_weights
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(self.training_transitions) // batch_size)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")

        print(f"✓ Neural Risk Model trained successfully\n")
        self.neural_risk_model.eval()

    def get_transition_risk(
        self,
        task_context,
        history,
        step_idx,
        system_prompts=None,
        is_initial=False,
        initial_state=None,
        initial_text=None,
        transition_state=None,
    ):
        if system_prompts is None:
            system_prompts = {}

        # Explicit initial-state scoring path used by threshold optimization.
        if is_initial:
            if initial_state is None or initial_text is None:
                return 0.0, None, "InitialState"
            try:
                with torch.no_grad():
                    initial_risk = self.initial_risk_model([initial_state], [initial_text]).item()
                return float(np.clip(initial_risk, 0.0, 1.0)), None, "InitialState"
            except Exception:
                return 0.0, None, "InitialState"

        # Optional direct transition payload path used by threshold optimization.
        if transition_state is not None:
            payload = transition_state
            if isinstance(payload, tuple) and len(payload) == 1 and isinstance(payload[0], tuple):
                payload = payload[0]
            if not isinstance(payload, tuple) or len(payload) != 4:
                return 0.0, None, "NoTransition"

            node_state, _curr_state_hint, prev_text_vec, curr_text_vec = payload
            task_context_text = self._task_context_to_text(task_context)
            task_text_vec = self.text_extractor.encode(task_context_text)

            try:
                with torch.no_grad():
                    neural_risk = float(
                        self.neural_risk_model.forward(
                            node_state,
                            task_text_vec,
                            prev_text_vec,
                            curr_text_vec,
                        )
                    )
            except Exception:
                neural_risk = self.global_hazard_rate

            transition_hazard = self._estimate_transition_hazard(node_state)
            mix = float(np.mean([float(neural_risk), transition_hazard]))
            return float(np.clip(mix, 0.0, 1.0)), None, node_state

        if history is None:
            history = []
        if step_idx < -1 or step_idx >= len(history) - 1:
            return 0.0, None, "NoTransition"

        # Backward-compatible initial transition path: task -> first message.
        if step_idx == -1:
            if not history:
                return 0.0, None, "NoTransition"
            next_msg = history[0]
            next_agent = self.normalize_agent_name(next_msg.get('name', next_msg.get('role', 'Unknown')))
            initial_text = self.text_extractor.encode(str(next_msg.get('content', '')))
            task_type = next_msg.get('task_target') if isinstance(next_msg, dict) else None
            if not task_type or task_type not in TaskType.all():
                try:
                    task_type = self.state_manager.classifier.classify_task_type(task_context)
                except Exception:
                    task_type = TaskType.OTHER

            try:
                with torch.no_grad():
                    initial_risk = self.initial_risk_model([(task_type, next_agent)], [initial_text]).item()
                return float(np.clip(initial_risk, 0.0, 1.0)), None, "InitialState"
            except Exception:
                return 0.0, None, "InitialState"

        curr_msg = history[step_idx]
        next_msg = history[step_idx + 1]
        curr_agent = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
        next_agent = next_msg.get('name', next_msg.get('role', 'Unknown'))
        curr_desc = self.resolve_agent_description(system_prompts, curr_agent)
        next_desc = self.resolve_agent_description(system_prompts, next_agent)

        curr_ctx = self._build_history_context(history, step_idx)
        next_ctx = self._build_history_context(history, step_idx + 1)

        curr_state = self.state_manager.extract_state(
            task_context,
            curr_msg,
            curr_desc,
            history_context=curr_ctx,
        )
        next_state = self.state_manager.extract_state(
            task_context,
            next_msg,
            next_desc,
            history_context=next_ctx,
        )

        (_p_role, p_action, p_consensus, p_txt) = curr_state
        (c_role, _c_action, _c_consensus, c_txt) = next_state
        node_state = (p_action, p_consensus, c_role)

        task_context_text = self._task_context_to_text(task_context)
        task_text_vec = self.text_extractor.encode(task_context_text)
        prev_text_vec = self.text_extractor.encode(p_txt)
        curr_text_vec = self.text_extractor.encode(c_txt)

        try:
            with torch.no_grad():
                neural_risk = float(
                    self.neural_risk_model.forward(
                        node_state,
                        task_text_vec,
                        prev_text_vec,
                        curr_text_vec,
                    )
                )
        except Exception:
            neural_risk = self.global_hazard_rate

        transition_hazard = self._estimate_transition_hazard(node_state)
        mix = float(np.mean([float(neural_risk), transition_hazard]))
        return float(np.clip(mix, 0.0, 1.0)), next_state, node_state

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
        "detection_with_correct_agent": 0,
        "peak_risk_recall_hits": 0,
    }
    
    RISK_THRESHOLD = markov_model.optimal_threshold
    print(f"Using Mean Transition Threshold: {RISK_THRESHOLD:.4f}")
    
    detect_ratios = []
    
    for idx, json_data in enumerate(dataset):
        history = json_data.get('history', [])
        total_steps = len(history)
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
        outcome = "MISSED"
        agent_match = False
        
        max_observed_risk = 0.0
        step_records = []

        # Unified loop: handle all transitions uniformly from step_idx=-1 (task->first) onwards.
        for i in range(-1, max(-1, total_steps - 1)):
            risk, next_state, transition_key = markov_model.get_transition_risk(
                task_context=task_context,
                history=history,
                step_idx=i,
                system_prompts=system_prompts,
            )

            if i == -1:
                target_idx = 0
                target_msg = history[0]
            else:
                target_idx = i + 1
                target_msg = history[target_idx]
            
            agent_name = target_msg.get('name', target_msg.get('role', 'Unknown'))
            risk = markov_model.calibrate_risk(risk, agent_name)
            step_records.append((target_idx, agent_name, risk))
            
            if target_idx == mistake_step:
                max_observed_risk = risk
            threshold = markov_model.get_transition_threshold(target_idx, total_steps, transition_key=transition_key)
            
            # Only detect if not yet detected.
            if detected_at == -1 and risk > threshold:
                detected_at = target_idx
                detected_agent = agent_name

        # New metric: peak-risk step recall over full trajectory.
        if step_records:
            peak_step, peak_agent, peak_risk = max(step_records, key=lambda x: x[2])
            if peak_step == mistake_step:
                stats["peak_risk_recall_hits"] += 1
        else:
            peak_step, peak_agent, peak_risk = -1, "UNKNOWN", 0.0
        
        if detected_at != -1:
            detected_norm = markov_model.normalize_agent_name(detected_agent)
            expected_norm = markov_model.normalize_agent_name(mistake_agent)
            agent_match = detected_norm == expected_norm
            if agent_match:
                stats["agent_hit"] += 1
            
            # Determine detection type and correctness
            if detected_at == mistake_step:
                stats["detected"] += 1
                if agent_match:
                    stats["detection_with_correct_agent"] += 1
                outcome = "EXACT_HIT"
            elif detected_at < mistake_step:
                stats["early_detection"] += 1
                if agent_match:
                    stats["detection_with_correct_agent"] += 1
                outcome = "EARLY_WARN"
            else:
                outcome = "LATE_MATCH"

            
            detect_ratios.append(detected_at / len(history))
            agent_status = "✓" if agent_match else "✗"
            print(f"[ID {idx}] {outcome} @ {detected_at} (Tgt: {mistake_step}) Agent: {detected_agent} {agent_status} (Expected: {mistake_agent})")
            print(f"           PeakRisk @ {peak_step} (risk={peak_risk:.4f}, agent={peak_agent})")

        else:
            detect_ratios.append(1.0)
            print(f"[ID {idx}] MISSED {mistake_step} (Risk @ Mistake Step: {max_observed_risk:.4f})")
            print(f"           PeakRisk @ {peak_step} (risk={peak_risk:.4f}, agent={peak_agent})")

    print("\n=== Final Results ===")
    print(f"Total Valid Cases: {stats['valid_cases']}")
    print(f"Exact Hit: {stats['detected']} ({stats['detected']/stats['valid_cases']:.2%})")
    print(f"Early Warning: {stats['early_detection']} ({stats['early_detection']/stats['valid_cases']:.2%})")
    combined = stats['detected'] + stats['early_detection']
    print(f"Combined Recall (Exact+Early): {combined/stats['valid_cases']:.2%}")
    print(
        f"Peak-Risk Step Recall: {stats['peak_risk_recall_hits']} "
        f"({stats['peak_risk_recall_hits']/stats['valid_cases']:.2%})"
    )
    print(f"\nAgent Hit Rate: {stats['agent_hit']} ({stats['agent_hit']/stats['valid_cases']:.2%})")
    print(f"Detection with Correct Agent: {stats['detection_with_correct_agent']} ({stats['detection_with_correct_agent']/stats['valid_cases']:.2%})")
    print(f"Avg Detection position ratio: {np.mean(detect_ratios):.4f}")


if __name__ == "__main__":
    datasets_dirs = [
        "Who&When/Algorithm-Generated",
        #"Who&When/Hand-Crafted",
        #"datasets/mmlu",
        #"datasets/aqua",
        #"datasets/humaneval"
    ]
    
    all_files = []
    dataset_counts = {}
    
    for data_dir in datasets_dirs:
        if os.path.exists(data_dir):
            files = glob.glob(os.path.join(data_dir, "*.json"))
            valid_files = [f for f in files if os.path.getsize(f) > 0]
            all_files.extend(valid_files)
            dataset_counts[data_dir] = len(valid_files)
            print(f"Loaded {len(valid_files)} files from {data_dir}")
        else:
            print(f"Warning: Directory {data_dir} not found")
    
    print(f"\nTotal files from all datasets: {len(all_files)}")
    print(f"Dataset breakdown: {dataset_counts}\n")

    random.shuffle(all_files)
    
    split = int(len(all_files) * 0.2)
    train_files = all_files[:split]
    test_files = all_files[split:]
    
    train_set = WhoWhenDataset(file_paths=train_files)
    test_set = WhoWhenDataset(file_paths=test_files)
    
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    classifier = LLMStateClassifier()
    
    markov = DiscreteStateMarkov(classifier)
    markov.fit(train_set)
    
    run_evaluation(markov, test_set)
