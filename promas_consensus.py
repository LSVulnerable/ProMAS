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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
    next_action is used only in loss as behavioral reference, not as model input.
    """
    def __init__(self, hidden_dim=128, text_dim=384, dropout=0.2):
        super().__init__()

        self.actions = {c: i for i, c in enumerate(ActionType.all())}
        self.consensuses = {c: i for i, c in enumerate(ConsensusState.all())}
        self.roles = {c: i for i, c in enumerate(AgentRole.all())}

        self.action_reference = {
            ActionType.OK: 0.05,
            ActionType.EVALUATION: 0.20,
            ActionType.PLANNING: 0.30,
            ActionType.TOOL_CALL: 0.40,
            ActionType.INFORMATION: 0.45,
            ActionType.ACT: 0.55,
            ActionType.FAIL: 0.95,
        }

        emb_dim = 8
        self.prev_action_emb = nn.Embedding(len(self.actions), emb_dim)
        self.consensus_emb = nn.Embedding(len(self.consensuses), emb_dim)
        self.next_role_emb = nn.Embedding(len(self.roles), emb_dim)

        self.text_dim = text_dim
        state_dim = (emb_dim * 3) + self.text_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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

    def compute_loss(self, node_states, task_texts, prev_texts, curr_texts, labels, next_actions, sample_weights=None):
        logits = []
        for n_state, t_txt, p_txt, c_txt in zip(node_states, task_texts, prev_texts, curr_texts):
            node_emb = self.encode_node_state(n_state, c_txt)
            node_encoded = self.node_encoder(node_emb)
            logic_feats = self._logic_features(t_txt, p_txt, c_txt)
            combined = torch.cat([node_encoded, logic_feats], dim=1)
            logits.append(self.risk_head(combined))

        logits = torch.cat(logits).squeeze(-1)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
        w_t = torch.tensor(sample_weights, dtype=torch.float32, device=self.device) if sample_weights is not None else None

        # Main risk supervision (success/failure transition labels).
        loss_vec = F.binary_cross_entropy_with_logits(logits, labels_t, reduction='none')
        if w_t is not None:
            loss_vec = loss_vec * w_t

        # Action is used only as training reference target, not model input.
        action_targets = torch.tensor(
            [self.action_reference.get(a, 0.5) for a in next_actions],
            dtype=torch.float32,
            device=self.device,
        )
        probs = torch.sigmoid(logits)
        action_ref_loss = F.mse_loss(probs, action_targets)

        # Contrastive separation: failures should have higher risk than successes.
        pos_logits = logits[labels_t >= 0.5]
        neg_logits = logits[labels_t < 0.5]
        contrastive = torch.tensor(0.0, device=self.device)
        if len(pos_logits) > 0 and len(neg_logits) > 0:
            margin = 0.25
            contrastive = F.relu(margin - (torch.mean(pos_logits) - torch.mean(neg_logits)))

        return loss_vec.mean() + (0.25 * action_ref_loss) + (0.20 * contrastive)

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
        self.optimal_threshold = 0.5
        self.transition_thresholds = {}
        self.transition_counts = defaultdict(int)
        self.transition_failure_sums = defaultdict(float)
        self.global_impact_sum = 0.0
        self.global_impact_count = 0
        self.global_hazard_rate = 0.5
        self.training_transitions = []

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
        effective_step = 0 if step_idx == -1 else step_idx + 1
        if mistake_step < 0:
            return 0.0
        return 1.0 if effective_step == mistake_step else 0.0

    def _cluster_boundary(self, success_scores, failure_scores):
        """
        Find threshold between success and failure distributions.
        Using weighted mean instead of k-means for simplicity and stability.
        """
        if not success_scores and not failure_scores:
            return float(self.optimal_threshold)
        if not success_scores:
            return float(np.clip(np.mean(failure_scores), 0.01, 0.99))
        if not failure_scores:
            return float(np.clip(np.mean(success_scores), 0.01, 0.99))

        success_mean = float(np.mean(np.array(success_scores, dtype=np.float32)))
        failure_mean = float(np.mean(np.array(failure_scores, dtype=np.float32)))
        
        # Weighted boundary: favor failure distribution (70%) to be more sensitive to failures
        threshold = (success_mean * 0.8) + (failure_mean * 0.2)
        return float(np.clip(threshold, 0.01, 0.99))

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

    def optimize_threshold(self, dataset):
        """Learn threshold from train-set score distributions."""
        print("\nOptimizing Detection Threshold...")
        success_scores = []
        failure_scores = []
        state_success_scores = defaultdict(list)
        state_failure_scores = defaultdict(list)

        limit_files = len(dataset)
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            total_steps = len(history)
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            system_prompts = json_data.get('system_prompt', {})

            for i in range(-1, max(-1, total_steps - 1)):
                risk, next_state, transition_key = self.get_transition_risk(
                    task_context=task_context,
                    history=history,
                    step_idx=i,
                    system_prompts=system_prompts,
                )

                risk = float(risk)
                is_failure = self._transition_label(i, mistake_step)
                if is_failure >= 0.5:
                    failure_scores.append(risk)
                    state_failure_scores[transition_key].append(risk)
                else:
                    success_scores.append(risk)
                    state_success_scores[transition_key].append(risk)

                _ = next_state

        self.optimal_threshold = self._cluster_boundary(success_scores, failure_scores)
        print(
            f"Stats: Mean Success Risk={np.mean(success_scores) if success_scores else 0.0:.4f}, "
            f"Mean Failure Risk={np.mean(failure_scores) if failure_scores else 0.0:.4f}"
        )
        print(f"Selected Threshold (Cluster Boundary): {self.optimal_threshold:.4f}")

        # State-specific thresholds via class-conditioned cluster boundaries.
        all_keys = set(state_success_scores.keys()) | set(state_failure_scores.keys())
        self.transition_thresholds = {}
        for key in all_keys:
            success_arr = state_success_scores.get(key, [])
            failure_arr = state_failure_scores.get(key, [])
            if len(success_arr) + len(failure_arr) >= 6:
                thr = self._cluster_boundary(success_arr, failure_arr)
            else:
                thr = float(self.optimal_threshold)
            self.transition_thresholds[key] = thr
        print(f"Learned State-Specific Thresholds: {len(self.transition_thresholds)}")

    def fit(self, dataset):
        print("Fitting Markov + Neural model with compact state...")
        limit_files = len(dataset)
        self.training_transitions = []
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
                    next_agent_name = next_msg.get('name', next_msg.get('role', 'Unknown'))
                    next_desc = self.resolve_agent_description(sys_prompts, next_agent_name)
                    try:
                        curr_state = self.state_manager.extract_task_state(task_context)
                        next_state = self.state_manager.extract_state(
                            task_context,
                            next_msg,
                            next_desc,
                            history_context="",
                        )
                    except Exception:
                        continue
                    effective_step = -1
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
        
        print(f"\n[Step 2/3] Training Neural Risk Model on {len(self.training_transitions)} transitions...")
        self._train_neural_model()
        
        self.optimize_threshold(dataset)

    def _train_neural_model(self, epochs=20, batch_size=32, lr=0.001):
        """Train neural transition risk scorer."""
        optimizer = torch.optim.Adam(self.neural_risk_model.parameters(), lr=lr)
        transitions = self.training_transitions
        if not transitions:
            print("No transitions to train.")
            return
        n_batches = len(transitions) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle transitions
            indices = np.random.permutation(len(transitions))
            
            for batch_idx in range(0, len(transitions), batch_size):
                batch_indices = indices[batch_idx:batch_idx+batch_size]
                
                node_states = [transitions[i][0] for i in batch_indices]
                task_texts = [transitions[i][1] for i in batch_indices]
                prev_texts = [transitions[i][2] for i in batch_indices]
                curr_texts = [transitions[i][3] for i in batch_indices]
                labels = [transitions[i][4] for i in batch_indices]
                next_actions = [transitions[i][5] for i in batch_indices]
                sample_weights = [transitions[i][6] for i in batch_indices]
                
                optimizer.zero_grad()
                loss = self.neural_risk_model.compute_loss(
                    node_states,
                    task_texts,
                    prev_texts,
                    curr_texts,
                    labels,
                    next_actions,
                    sample_weights=sample_weights,
                )
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        print(f"✓ Neural Risk Model trained successfully\n")
        self.neural_risk_model.eval()

    def get_transition_risk(self, task_context, history, step_idx, system_prompts=None):
        if system_prompts is None:
            system_prompts = {}
        if step_idx < -1 or step_idx >= len(history) - 1:
            return 0.0, None, "NoTransition"

        if step_idx == -1:
            if not history:
                return 0.0, None, "NoTransition"
            next_msg = history[0]
            next_agent = next_msg.get('name', next_msg.get('role', 'Unknown'))
            next_desc = self.resolve_agent_description(system_prompts, next_agent)
            curr_state = self.state_manager.extract_task_state(task_context)
            next_state = self.state_manager.extract_state(
                task_context,
                next_msg,
                next_desc,
                history_context="",
            )
        else:
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

        (p_role, p_action, p_consensus, p_txt) = curr_state
        (c_role, c_action, c_consensus, c_txt) = next_state
        node_state = (p_action, p_consensus, c_role)

        task_context_text = self._task_context_to_text(task_context)
        task_text_vec = self.text_extractor.encode(task_context_text)
        prev_text_vec = self.text_extractor.encode(p_txt)
        curr_text_vec = self.text_extractor.encode(c_txt)

        try:
            with torch.no_grad():
                neural_logit = self.neural_risk_model.forward(
                    node_state,
                    task_text_vec,
                    prev_text_vec,
                    curr_text_vec,
                )
                neural_risk = float(torch.sigmoid(torch.tensor(neural_logit)).item())
        except Exception:
            neural_risk = self.global_hazard_rate

        transition_key = node_state
        transition_hazard = self._estimate_transition_hazard(transition_key)
        mix = float(np.mean([float(neural_risk), transition_hazard]))

        return float(np.clip(mix, 0.0, 1.0)), next_state, transition_key

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
        
        limit = min(len(history), mistake_step + 5)
        
        max_observed_risk = 0.0
        step_records = []

        # Unified loop: handle all transitions uniformly from step_idx=-1 (task→first) onwards
        for i in range(-1, max(-1, limit - 1)):
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

        else:
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


if __name__ == "__main__":
    datasets_dirs = [
        #"Who&When/Algorithm-Generated",
        #"Who&When/Hand-Crafted",
        "datasets/mmlu",
        "datasets/aqua",
        "datasets/humaneval"
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
    
    split = int(len(all_files) * 0.4)
    train_files = all_files[:split]
    test_files = all_files[split:]
    
    train_set = WhoWhenDataset(file_paths=train_files)
    test_set = WhoWhenDataset(file_paths=test_files)
    
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    classifier = LLMStateClassifier()
    
    markov = DiscreteStateMarkov(classifier)
    markov.fit(train_set)
    
    run_evaluation(markov, test_set)
