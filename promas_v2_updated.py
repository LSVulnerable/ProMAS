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

class ClaimState:
    NOVEL_CLAIM = "NOVEL_CLAIM"
    NO_CLAIM = "NO_CLAIM"

    @classmethod
    def all(cls):
        return [cls.NOVEL_CLAIM, cls.NO_CLAIM]

class EvidenceState:
    HAS_EVIDENCE = "HAS_EVIDENCE"
    NO_EVIDENCE = "NO_EVIDENCE"

    @classmethod
    def all(cls):
        return [cls.HAS_EVIDENCE, cls.NO_EVIDENCE]

class PeerStance:
    AGREE = "AGREE"
    CHALLENGE = "CHALLENGE"
    IGNORE = "IGNORE"
    NEUTRAL = "NEUTRAL"
    
    @classmethod
    def all(cls):
        return [cls.AGREE, cls.CHALLENGE, cls.IGNORE, cls.NEUTRAL]

class ConsensusTrajectory:
    FRAGMENTED = "FRAGMENTED"
    EMERGING = "EMERGING"
    STABLE = "STABLE"
    BLIND = "BLIND"

    @classmethod
    def all(cls):
        return [cls.FRAGMENTED, cls.EMERGING, cls.STABLE, cls.BLIND]

class EvidenceTrajectory:
    SPARSE = "SPARSE"
    MIXED = "MIXED"
    GROUNDED = "GROUNDED"

    @classmethod
    def all(cls):
        return [cls.SPARSE, cls.MIXED, cls.GROUNDED]

class TaskTrajectory:
    FRAMING = "FRAMING"
    HYPOTHESIS = "HYPOTHESIS"
    SOLVING = "SOLVING"
    VERIFYING = "VERIFYING"
    FINALIZING = "FINALIZING"

    @classmethod
    def all(cls):
        return [cls.FRAMING, cls.HYPOTHESIS, cls.SOLVING, cls.VERIFYING, cls.FINALIZING]

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

    def classify_state(self, task_context, history_item, agent_description="", history_context=""):
        # Handle dict or string task_context (humaneval has dict, others have string)
        if isinstance(task_context, dict):
            task_context_str = task_context.get('prompt', str(task_context))
        else:
            task_context_str = str(task_context)
        
        content = history_item.get('content', '') or ''  # Handle None values
        if content:
            content = content[:1000]
        agent_name = history_item.get('name', history_item.get('role', 'Unknown'))
        
        # Trajectory-aware prompt: output a system-level discussion state, not only utterance tags
        system_prompt = (
            "You are a trajectory-level epistemic compressor for multi-agent problem solving.\\n"
            "Analyze history + current utterance, then infer: current global discussion state after this turn.\\n"
            "Do NOT rely on speaker identity/role labels; prioritize task progress, evidence quality, and consensus dynamics.\\n"
            "Output fields:\\n"
            "1) claim_state: NOVEL_CLAIM or NO_CLAIM\\n"
            "2) evidence_state: HAS_EVIDENCE or NO_EVIDENCE\\n"
            "3) peer_stance: AGREE / CHALLENGE / IGNORE / NEUTRAL\\n"
            "4) consensus_trajectory: FRAGMENTED / EMERGING / STABLE / BLIND\\n"
            "5) evidence_trajectory: SPARSE / MIXED / GROUNDED\\n"
            "6) task_trajectory: FRAMING / HYPOTHESIS / SOLVING / VERIFYING / FINALIZING\\n"
            "\\n"
            "Output ONLY valid JSON with exact keys and values:\\n"
            "{\\\"claim_state\\\": \\\"NOVEL_CLAIM|NO_CLAIM\\\", \\\"evidence_state\\\": \\\"HAS_EVIDENCE|NO_EVIDENCE\\\", \\\"peer_stance\\\": \\\"AGREE|CHALLENGE|IGNORE|NEUTRAL\\\", \\\"consensus_trajectory\\\": \\\"FRAGMENTED|EMERGING|STABLE|BLIND\\\", \\\"evidence_trajectory\\\": \\\"SPARSE|MIXED|GROUNDED\\\", \\\"task_trajectory\\\": \\\"FRAMING|HYPOTHESIS|SOLVING|VERIFYING|FINALIZING\\\"}"
        )
        
        ctx_text = f"Prior context: {history_context[:700]}\\n" if history_context else ""
        
        user_prompt = (
            f"Task Context: {task_context_str[:100]}\\n"
            f"Speaker: {agent_name}\\n"
            f"{ctx_text}"
            f"Current Message: {content}\\n\\n"
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
            
        claim = data.get("claim_state", ClaimState.NO_CLAIM)
        evidence = data.get("evidence_state", EvidenceState.NO_EVIDENCE)
        stance = data.get("peer_stance", PeerStance.NEUTRAL)
        consensus_traj = data.get("consensus_trajectory", ConsensusTrajectory.FRAGMENTED)
        evidence_traj = data.get("evidence_trajectory", EvidenceTrajectory.SPARSE)
        task_traj = data.get("task_trajectory", TaskTrajectory.FRAMING)
        
        # Validate and fixup
        if claim not in ClaimState.all(): claim = ClaimState.NO_CLAIM
        if evidence not in EvidenceState.all(): evidence = EvidenceState.NO_EVIDENCE
        if stance not in PeerStance.all(): stance = PeerStance.NEUTRAL
        if consensus_traj not in ConsensusTrajectory.all():
            consensus_traj = ConsensusTrajectory.FRAGMENTED
        if evidence_traj not in EvidenceTrajectory.all():
            evidence_traj = EvidenceTrajectory.SPARSE
        if task_traj not in TaskTrajectory.all():
            task_traj = TaskTrajectory.FRAMING
        
        return claim, evidence, stance, consensus_traj, evidence_traj, task_traj

class PreDefinedStateManager:
    """
    Wrapper that now uses LLM Classifier instead of if-else.
    Includes caching to avoid re-running LLM on same identical messages.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.cache = {}
    
    def extract_state(self, task_context, history_item, has_prior_error=False, agent_description="", history_context=""):
        # Create a cache key from content hash + agent
        # Ensure task_context is a string (humaneval may have dict, others have string)
        if isinstance(task_context, dict):
            task_context_str = task_context.get('prompt', str(task_context))
        else:
            task_context_str = str(task_context)
        
        content_hash = hash(history_item.get('content', ''))
        key = (task_context_str[:50], content_hash, hash(history_context[:240]))
        
        if key in self.cache:
            claim, evidence, stance, consensus_traj, evidence_traj, task_traj = self.cache[key]
        else:
            claim, evidence, stance, consensus_traj, evidence_traj, task_traj = self.classifier.classify_state(
                task_context,
                history_item,
                agent_description=agent_description,
                history_context=history_context
            )
            self.cache[key] = (claim, evidence, stance, consensus_traj, evidence_traj, task_traj)
            
        content_text = history_item.get('content', '') or ''
        return (claim, evidence, stance, consensus_traj, evidence_traj, task_traj, has_prior_error, content_text)

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
    Uses Context and Auxiliary Task to learn risk scoring based on Epistemic States.
    """
    def __init__(self, hidden_dim=128, text_dim=384, dropout=0.2, focal_alpha=0.75, focal_gamma=2.0, aux_stance_weight=0.1):
        super().__init__()
        
        self.claims = {c: i for i, c in enumerate(ClaimState.all())}
        self.evidences = {e: i for i, e in enumerate(EvidenceState.all())}
        self.stances = {s: i for i, s in enumerate(PeerStance.all())}
        self.consensus_states = {c: i for i, c in enumerate(ConsensusTrajectory.all())}
        self.evidence_states = {e: i for i, e in enumerate(EvidenceTrajectory.all())}
        self.task_states = {t: i for i, t in enumerate(TaskTrajectory.all())}
        
        self.claim_emb_dim = 4
        self.evidence_emb_dim = 4
        self.stance_emb_dim = 4
        self.consensus_emb_dim = 4
        self.evidence_state_emb_dim = 3
        self.task_state_emb_dim = 4
        
        self.claim_emb = nn.Embedding(len(self.claims), self.claim_emb_dim)
        self.evidence_emb = nn.Embedding(len(self.evidences), self.evidence_emb_dim)
        self.stance_emb = nn.Embedding(len(self.stances), self.stance_emb_dim)
        self.consensus_emb = nn.Embedding(len(self.consensus_states), self.consensus_emb_dim)
        self.evidence_state_emb = nn.Embedding(len(self.evidence_states), self.evidence_state_emb_dim)
        self.task_state_emb = nn.Embedding(len(self.task_states), self.task_state_emb_dim)
        
        self.text_dim = text_dim
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_stance_weight = aux_stance_weight
        
        state_dim = (
            self.claim_emb_dim
            + self.evidence_emb_dim
            + self.stance_emb_dim
            + self.consensus_emb_dim
            + self.evidence_state_emb_dim
            + self.task_state_emb_dim
            + self.text_dim
        )
        
        self.prev_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.curr_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        combined_dim = (hidden_dim * 2) + 2  # + 2 logic features
        
        # 1. Main Task: Predict Risk
        self.risk_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        
        # 2. Auxiliary Task: Predict Current Stance (instead of Action)
        self.stance_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.stances))
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

    def _logic_features(self, prev_text, curr_text):
        prev_t = self._to_text_tensor(prev_text)
        curr_t = self._to_text_tensor(curr_text)
        cos_sim = F.cosine_similarity(prev_t, curr_t, dim=1, eps=1e-8).unsqueeze(1)
        semantic_delta = torch.mean(torch.abs(curr_t - prev_t), dim=1, keepdim=True)
        feats = torch.cat([cos_sim, semantic_delta], dim=1)
        return feats * 1.5

    def encode_state(self, state_tuple, text_vector):
        claim, evidence, stance, consensus_traj, evidence_traj, task_traj = state_tuple
        
        c_idx = self.claims.get(claim, 0)
        e_idx = self.evidences.get(evidence, 0)
        s_idx = self.stances.get(stance, 0)
        cs_idx = self.consensus_states.get(consensus_traj, 0)
        es_idx = self.evidence_states.get(evidence_traj, 0)
        ts_idx = self.task_states.get(task_traj, 0)
        
        c_t = torch.tensor([c_idx], dtype=torch.long, device=self.device)
        e_t = torch.tensor([e_idx], dtype=torch.long, device=self.device)
        s_t = torch.tensor([s_idx], dtype=torch.long, device=self.device)
        cs_t = torch.tensor([cs_idx], dtype=torch.long, device=self.device)
        es_t = torch.tensor([es_idx], dtype=torch.long, device=self.device)
        ts_t = torch.tensor([ts_idx], dtype=torch.long, device=self.device)
        
        text_t = self._to_text_tensor(text_vector)
        
        state_emb = torch.cat(
            [
                self.claim_emb(c_t),
                self.evidence_emb(e_t),
                self.stance_emb(s_t),
                self.consensus_emb(cs_t),
                self.evidence_state_emb(es_t),
                self.task_state_emb(ts_t),
                text_t,
            ],
            dim=1,
        )
        return state_emb

    def forward(self, prev_state, curr_state, prev_text, curr_text):
        """
        Forward pass for risk scoring.
        Returns: (risk_score: float, stance_logits: tensor)
        """
        prev_emb = self.encode_state(prev_state, prev_text)
        curr_emb = self.encode_state(curr_state, curr_text)
        
        prev_encoded = self.prev_encoder(prev_emb)  
        curr_encoded = self.curr_encoder(curr_emb)  
        logic_feats = self._logic_features(prev_text, curr_text)
        
        combined = torch.cat([prev_encoded, curr_encoded, logic_feats], dim=1)  
        
        risk = self.risk_head(combined) 
        stance_logits = self.stance_head(combined)
        
        # Heuristic boost: NOVEL_CLAIM + NO_EVIDENCE in prev, and AGREE in curr -> risky
        prev_claim, prev_evidence, prev_stance, prev_consensus, _, _ = prev_state
        _, _, curr_stance, curr_consensus, _, _ = curr_state
        
        # Check for dangerous pattern: [NOVEL_CLAIM, NO_EVIDENCE, *] -> [*, *, AGREE]
        if (prev_claim == "NOVEL_CLAIM" and prev_evidence == "NO_EVIDENCE"):
            if curr_stance == "AGREE":
                # This is the high-risk signature: unsupported claim accepted by peer
                risk = risk * 1.35 + 0.12
        if prev_consensus == ConsensusTrajectory.EMERGING and curr_consensus == ConsensusTrajectory.BLIND:
            risk = risk + 0.25
        
        return risk.squeeze().item(), stance_logits

    def focal_bce_loss(self, probs, labels, sample_weights=None):
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(probs, labels, reduction='none')
        pt = torch.where(labels == 1, probs, 1 - probs)
        alpha_t = torch.where(labels == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal = alpha_t * ((1 - pt) ** self.focal_gamma) * bce
        if sample_weights is not None:
            focal = focal * sample_weights
        return focal.mean()

    def compute_loss(self, prev_states, curr_states, prev_texts, curr_texts, labels, target_stances, sample_weights=None):
        risks, stance_preds = [], []
        for p_state, c_state, p_txt, c_txt in zip(prev_states, curr_states, prev_texts, curr_texts):
            prev_emb = self.encode_state(p_state, p_txt)
            curr_emb = self.encode_state(c_state, c_txt)
            logic_feats = self._logic_features(p_txt, c_txt)
            
            combined = torch.cat([self.prev_encoder(prev_emb), self.curr_encoder(curr_emb), logic_feats], dim=1)
            risks.append(self.risk_head(combined))
            stance_preds.append(self.stance_head(combined))
            
        risks = torch.cat(risks).squeeze(-1) 
        stance_preds = torch.cat(stance_preds, dim=0) 
        labels_t = torch.tensor(labels, dtype=torch.float32, device=self.device)
        
        target_stance_idx = [self.stances.get(s, 0) for s in target_stances]
        target_stances_t = torch.tensor(target_stance_idx, dtype=torch.long, device=self.device)
        w_t = torch.tensor(sample_weights, dtype=torch.float32, device=self.device) if sample_weights is not None else None
        
        loss_risk = self.focal_bce_loss(risks, labels_t, sample_weights=w_t)
        loss_stance = nn.CrossEntropyLoss()(stance_preds, target_stances_t)
        
        return loss_risk + self.aux_stance_weight * loss_stance

class DiscreteStateMarkov:
    """
    Hybrid Model:
    1. Hierarchical Markov Chain (for statistical priors)
    2. Neural Risk Model (for learned risk scoring via contrastive learning)
    3. Transition-first risk scoring (no special-cased initial model)
    """
    def __init__(self, classifier):
        self.state_manager = PreDefinedStateManager(classifier)
        self.text_extractor = TextFeatureExtractor()
        
        # Neural Risk Model
        self.neural_risk_model = NeuralRiskModel(hidden_dim=128, text_dim=self.text_extractor.dim)
        self.optimal_threshold = 0.1
        self.agent_bias = {}
        self.use_agent_bias = False
        self.agent_bias_cap = 0.18
        self.agent_bias_min_support = 5
        self.hazard_horizon = 3  # Increased from 1: allows [mistake_step-3, mistake_step] to be hazardous
        self.min_detection_step = 2
        
        # Training data for neural model
        self.training_transitions = []  # list of (prev_state, curr_state, prev_text, curr_text, is_mistake, curr_action, weight)

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

    def _text_info_density(self, text):
        content = str(text or "")
        if not content.strip():
            return 0.0
        lowered = content.lower()
        tokens = re.findall(r"[a-zA-Z0-9_]+", lowered)
        if not tokens:
            return 0.0
        uniq_ratio = len(set(tokens)) / max(1, len(tokens))
        evidence_markers = [
            "because", "therefore", "proof", "derive", "calculation", "equation",
            "tool", "search", "result", "observation", "data", "verify", "let"
        ]
        marker_hits = sum(1 for m in evidence_markers if m in lowered)
        digit_bonus = 1.0 if re.search(r"\d", content) else 0.0
        code_bonus = 1.0 if ("```" in content or "def " in content or "import " in content) else 0.0
        raw = 0.45 * uniq_ratio + 0.12 * marker_hits + 0.12 * digit_bonus + 0.10 * code_bonus
        return float(np.clip(raw, 0.0, 1.0))

    def _epistemic_risk(self, prev_state, curr_state, curr_text, step_idx, total_steps):
        c_claim, c_evidence, c_stance, c_consensus, c_evidence_traj, c_task = curr_state
        progress = float(step_idx) / max(1, total_steps - 1)
        info_density = self._text_info_density(curr_text)

        novelty_risk = 1.0 if (c_claim == ClaimState.NOVEL_CLAIM and c_evidence == EvidenceState.NO_EVIDENCE) else 0.0
        agree_risk = 0.0
        if prev_state is not None:
            p_claim, p_evidence, _, p_consensus, _, _ = prev_state
            if c_stance == PeerStance.AGREE and p_claim == ClaimState.NOVEL_CLAIM and p_evidence == EvidenceState.NO_EVIDENCE:
                agree_risk = 0.5
            if p_consensus == ConsensusTrajectory.EMERGING and c_consensus == ConsensusTrajectory.BLIND:
                agree_risk += 0.4

        challenge_bonus = -0.08 if c_stance == PeerStance.CHALLENGE else 0.0
        ignore_risk = 0.08 if c_stance == PeerStance.IGNORE else 0.0
        low_info_penalty = 0.35 * max(0.0, 0.6 - info_density)
        consensus_risk = 0.40 if c_consensus == ConsensusTrajectory.BLIND else 0.0
        sparse_evidence_risk = 0.30 if c_evidence_traj == EvidenceTrajectory.SPARSE else 0.0
        finalize_bonus = 0.16 if c_task == TaskTrajectory.FINALIZING else 0.0

        stage_weight = 0.35 + 0.65 * progress
        raw = (
            0.28 * novelty_risk
            + 0.34 * agree_risk
            + ignore_risk
            + low_info_penalty
            + consensus_risk
            + sparse_evidence_risk
            + finalize_bonus
            + challenge_bonus
        )
        return float(np.clip(raw * stage_weight, 0.0, 1.0)), info_density, progress

    def get_transition_threshold(self, step_idx, total_steps):
        return float(self.optimal_threshold)

    def _is_hazard_positive(self, step_idx, mistake_step):
        if mistake_step < 0:
            return False
        return (step_idx <= mistake_step) and (step_idx >= mistake_step - self.hazard_horizon)

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

    def calibrate_risk(self, risk, agent_name):
        if not self.use_agent_bias:
            return float(np.clip(float(risk), 0.0, 1.0))
        key = self.normalize_agent_name(agent_name)
        bias = float(self.agent_bias.get(key, 0.0))
        value = float(risk) - bias
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def predict_blame_agent(self, step_records, trigger_idx, window=2):
        if not step_records:
            return "UNKNOWN"
        trigger_pos = None
        for pos, (idx_j, _, _) in enumerate(step_records):
            if idx_j == trigger_idx:
                trigger_pos = pos
                break
        if trigger_pos is None:
            trigger_pos = min(range(len(step_records)), key=lambda k: abs(step_records[k][0] - trigger_idx))

        window = 1
        start = max(0, trigger_pos - window)
        end = min(len(step_records) - 1, trigger_pos + window)
        score_by_agent = defaultdict(float)
        name_by_agent = {}

        for j in range(start, end + 1):
            idx_j, agent_j, risk_j = step_records[j]
            agent_norm = self.normalize_agent_name(agent_j)
            name_by_agent[agent_norm] = agent_j
            if j == 0:
                score_by_agent[agent_norm] += 0.02 * float(risk_j)
                continue
            prev_risk = float(step_records[j - 1][2])
            delta = float(risk_j) - prev_risk
            if delta > 0:
                score_by_agent[agent_norm] += (1.8 * delta) + (0.12 * float(risk_j))
            else:
                score_by_agent[agent_norm] += 0.01 * float(risk_j)

        if score_by_agent:
            best_norm = max(score_by_agent.items(), key=lambda kv: kv[1])[0]
            return name_by_agent.get(best_norm, step_records[trigger_pos][1])
        return step_records[trigger_pos][1]

    def _simulate_detection(self, risks, mistake_step, total_steps):
        detected_at = -1
        for idx, risk in enumerate(risks):
            effective_idx = idx + 1
            if effective_idx < self.min_detection_step:
                continue
            threshold = self.get_transition_threshold(effective_idx, total_steps)
            if risk > threshold:
                detected_at = effective_idx
                break

        if detected_at == -1:
            return "MISSED"
        if detected_at == mistake_step:
            return "EXACT_HIT"
        if detected_at < mistake_step:
            return "EARLY_WARN"
        return "LATE_MATCH"

    def optimize_threshold(self, dataset):
        """"
        Find the best threshold that separates Safe steps from Mistake steps.
        """
        print("\nOptimizing Detection Threshold...")
        safe_scores = []
        mistake_scores = []
        raw_records = []
        case_risk_records = []
        mistake_steps_train = []
        raw_safe_agent = defaultdict(list)
        raw_mistake_agent = defaultdict(list)
        self.agent_bias = {}

        limit_files = len(dataset)
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            total_steps = len(history)
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            system_prompts = json_data.get('system_prompt', {})
            case_step_risks = []

            has_prior_error = False

            for i in range(max(0, total_steps - 1)):
                risk, next_state, _ = self.get_transition_risk(
                    task_context=task_context,
                    history=history,
                    step_idx=i,
                    has_prior_error=has_prior_error,
                    system_prompts=system_prompts,
                )

                next_msg = history[i + 1]
                next_agent_name = next_msg.get('name', next_msg.get('role', 'Unknown'))
                idx_effective = i + 1

                risk = float(risk)
                agent_norm = self.normalize_agent_name(next_agent_name)
                raw_records.append((risk, agent_norm, idx_effective, mistake_step))
                case_step_risks.append(risk)

                if self._is_hazard_positive(idx_effective, mistake_step):
                    raw_mistake_agent[agent_norm].append(risk)
                else:
                    raw_safe_agent[agent_norm].append(risk)

                _ = next_state

            if mistake_step >= 0 and case_step_risks:
                case_risk_records.append((case_step_risks, mistake_step, total_steps))
                mistake_steps_train.append(mistake_step)

        if mistake_steps_train:
            p15 = int(np.percentile(np.array(mistake_steps_train), 15))
            self.min_detection_step = max(1, p15)
        else:
            self.min_detection_step = 1

        all_safe_vals = []
        for vals in raw_safe_agent.values():
            all_safe_vals.extend(vals)
        global_safe_mean = float(np.mean(all_safe_vals)) if all_safe_vals else 0.0

        for agent_norm, safe_vals in raw_safe_agent.items():
            if len(safe_vals) < self.agent_bias_min_support:
                continue
            safe_mean = float(np.mean(safe_vals))
            raw_bias = safe_mean - global_safe_mean
            if raw_bias <= 0.0:
                continue
            mistake_vals = raw_mistake_agent.get(agent_norm, [])
            if len(mistake_vals) >= 3:
                sep = float(np.mean(mistake_vals)) - safe_mean
                if sep >= 0.07:
                    raw_bias *= 0.35
                elif sep >= 0.03:
                    raw_bias *= 0.60
            self.agent_bias[agent_norm] = float(np.clip(raw_bias, 0.0, self.agent_bias_cap))

        for raw_risk, agent_norm, step_idx, mistake_step in raw_records:
            risk = self.calibrate_risk(raw_risk, agent_norm)
            if self._is_hazard_positive(step_idx, mistake_step):
                mistake_scores.append(risk)
            else:
                safe_scores.append(risk)

        if mistake_scores:
            avg_safe = np.mean(safe_scores) if safe_scores else 0.0
            avg_mistake = np.mean(mistake_scores)
            std_safe = np.std(safe_scores) if safe_scores else 0.0

            all_scores = safe_scores + mistake_scores
            if all_scores and case_risk_records:
                safe_arr = np.array(safe_scores, dtype=np.float32) if safe_scores else np.array([0.0], dtype=np.float32)
                mist_arr = np.array(mistake_scores, dtype=np.float32) if mistake_scores else np.array([0.0], dtype=np.float32)
                base_th = 0.62 * float(np.percentile(safe_arr, 85)) + 0.38 * float(np.percentile(mist_arr, 35))

                candidates = np.unique(
                    np.clip(
                        np.array([base_th - 0.08, base_th - 0.04, base_th, base_th + 0.04, base_th + 0.08]),
                        0.01,
                        0.99,
                    )
                ).tolist()

                best_obj = -1e9
                best_th = float(base_th)
                for th in candidates:
                    self.optimal_threshold = float(th)
                    exact_cnt, early_cnt, miss_cnt, lag_penalty = 0, 0, 0, 0.0
                    for risks_case, ms_case, total_case in case_risk_records:
                        detected = -1
                        for j, r in enumerate(risks_case):
                            effective_j = j + 1
                            if effective_j < self.min_detection_step:
                                continue
                            if r > th:
                                detected = effective_j
                                break
                        if detected == -1:
                            miss_cnt += 1
                            continue
                        if detected == ms_case:
                            exact_cnt += 1
                        elif detected < ms_case:
                            early_cnt += 1
                        else:
                            lag_penalty += (detected - ms_case)

                    total_cnt = max(1, len(case_risk_records))
                    exact_rate = exact_cnt / total_cnt
                    early_rate = early_cnt / total_cnt
                    miss_rate = miss_cnt / total_cnt
                    norm_lag = lag_penalty / (total_cnt + 1e-6)
                    objective = (2.9 * exact_rate) - (1.4 * early_rate) - (1.2 * miss_rate) - (0.08 * norm_lag)
                    if objective > best_obj:
                        best_obj = objective
                        best_th = float(th)

                self.optimal_threshold = float(np.clip(best_th, 0.01, 0.99))
            else:
                self.optimal_threshold = avg_safe + 1.0 * std_safe

            print(f"Stats: Avg Safe Risk={avg_safe:.4f} (std={std_safe:.4f}), Avg Mistake Risk={avg_mistake:.4f}")
            print(f"Selected Optimal Threshold (Exact-oriented): {self.optimal_threshold:.4f}")
            print(f"Learned Min Detection Step: {self.min_detection_step}")
            if self.agent_bias:
                top_bias = sorted(self.agent_bias.items(), key=lambda kv: kv[1], reverse=True)[:5]
                print("Top agent bias calibration:")
                for agent_norm, bias in top_bias:
                    print(f"  {agent_norm}: -{bias:.4f}")

    def fit(self, dataset):
        print("Fitting Hybrid Markov Model with Neural Risk Learning...")
        limit_files = len(dataset)
        
        # Step 1: Collect training transitions
        print("\n[Step 1/3] Collecting training transitions...")
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            total_steps = len(history)
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            sys_prompts = json_data.get('system_prompt', {})

            mistake_claim = None
            if 0 <= mistake_step < len(history):
                m_msg = history[mistake_step]
                m_name = m_msg.get('name', m_msg.get('role', 'Unknown'))
                m_desc = sys_prompts.get(m_name, "")
                try:
                    m_history_ctx = self._build_history_context(history, mistake_step)
                    m_state = self.state_manager.extract_state(
                        task_context,
                        m_msg,
                        has_prior_error=False,
                        agent_description=m_desc,
                        history_context=m_history_ctx
                    )
                    mistake_claim = m_state[0]
                except:
                    mistake_claim = None
            
            has_prior_error = False
            
            for i in range(max(0, len(history) - 1)):
                curr_msg = history[i]
                next_msg = history[i + 1]
                curr_agent_name = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
                next_agent_name = next_msg.get('name', next_msg.get('role', 'Unknown'))
                curr_desc = sys_prompts.get(curr_agent_name, "")
                next_desc = sys_prompts.get(next_agent_name, "")

                try:
                    curr_history_ctx = self._build_history_context(history, i)
                    next_history_ctx = self._build_history_context(history, i + 1)
                    curr_state = self.state_manager.extract_state(
                        task_context,
                        curr_msg,
                        has_prior_error,
                        curr_desc,
                        history_context=curr_history_ctx,
                    )
                    next_state = self.state_manager.extract_state(
                        task_context,
                        next_msg,
                        has_prior_error,
                        next_desc,
                        history_context=next_history_ctx,
                    )
                except:
                    continue

                (p_claim, p_evidence, p_stance, p_consensus, p_e_traj, p_task, _, p_txt) = curr_state
                (c_claim, c_evidence, c_stance, c_consensus, c_e_traj, c_task, _, c_txt) = next_state

                prev_neural_state = (p_claim, p_evidence, p_stance, p_consensus, p_e_traj, p_task)
                curr_neural_state = (c_claim, c_evidence, c_stance, c_consensus, c_e_traj, c_task)

                prev_text_vec = self.text_extractor.encode(p_txt)
                curr_text_vec = self.text_extractor.encode(c_txt)
                effective_step = i + 1
                is_mistake = self._is_hazard_positive(effective_step, mistake_step)

                hard_weight = 1.0
                if not is_mistake and mistake_step >= 0:
                    dist = abs(effective_step - mistake_step)
                    if dist <= 2:
                        hard_weight += 1.0
                    if dist <= 1:
                        hard_weight += 0.5
                    if mistake_claim is not None and c_claim == mistake_claim and c_claim == ClaimState.NOVEL_CLAIM:
                        hard_weight += 0.15

                self.training_transitions.append(
                    (
                        prev_neural_state,
                        curr_neural_state,
                        prev_text_vec,
                        curr_text_vec,
                        int(is_mistake),
                        c_stance,
                        hard_weight,
                    )
                )

                _ = c_stance
        
        # Step 2: Train neural risk model with contrastive learning
        print(f"\n[Step 2/3] Training Neural Risk Model on {len(self.training_transitions)} transitions...")
        self._train_neural_model()
        
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
                prev_texts = [transitions[i][2] for i in batch_indices]
                curr_texts = [transitions[i][3] for i in batch_indices]
                labels = [transitions[i][4] for i in batch_indices]
                target_actions = [transitions[i][5] for i in batch_indices]
                sample_weights = [transitions[i][6] for i in batch_indices]
                
                optimizer.zero_grad()
                loss = self.neural_risk_model.compute_loss(prev_states, curr_states, prev_texts, curr_texts, labels, target_actions, sample_weights=sample_weights)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        print(f"✓ Neural Risk Model trained successfully\n")
        self.neural_risk_model.eval()

    def get_transition_risk(self, task_context, history, step_idx, has_prior_error=False, system_prompts=None):
        if system_prompts is None:
            system_prompts = {}
        if step_idx < 0 or step_idx >= len(history) - 1:
            return 0.0, None, "NoTransition"

        curr_msg = history[step_idx]
        next_msg = history[step_idx + 1]
        curr_agent = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
        next_agent = next_msg.get('name', next_msg.get('role', 'Unknown'))
        curr_desc = system_prompts.get(curr_agent, "")
        next_desc = system_prompts.get(next_agent, "")

        curr_ctx = self._build_history_context(history, step_idx)
        next_ctx = self._build_history_context(history, step_idx + 1)

        curr_state = self.state_manager.extract_state(
            task_context,
            curr_msg,
            has_prior_error,
            curr_desc,
            history_context=curr_ctx,
        )
        next_state = self.state_manager.extract_state(
            task_context,
            next_msg,
            has_prior_error,
            next_desc,
            history_context=next_ctx,
        )

        (p_claim, p_evidence, p_stance, p_consensus, p_e_traj, p_task, _, p_txt) = curr_state
        (c_claim, c_evidence, c_stance, c_consensus, c_e_traj, c_task, _, c_txt) = next_state

        prev_neural_state = (p_claim, p_evidence, p_stance, p_consensus, p_e_traj, p_task)
        curr_neural_state = (c_claim, c_evidence, c_stance, c_consensus, c_e_traj, c_task)

        prev_text_vec = self.text_extractor.encode(p_txt)
        curr_text_vec = self.text_extractor.encode(c_txt)

        try:
            with torch.no_grad():
                neural_risk, _ = self.neural_risk_model.forward(
                    prev_neural_state,
                    curr_neural_state,
                    prev_text_vec,
                    curr_text_vec,
                )
        except Exception:
            neural_risk = 0.35

        ep_risk, info_density, progress = self._epistemic_risk(
            prev_neural_state,
            curr_neural_state,
            c_txt,
            step_idx + 1,
            len(history),
        )

        mix = 0.55 * float(neural_risk) + 0.45 * ep_risk
        if info_density < 0.25 and curr_neural_state[2] == PeerStance.AGREE:
            mix += 0.10
        if curr_neural_state[3] == ConsensusTrajectory.BLIND:
            mix += 0.08
        if progress < 0.12:
            mix *= 0.85

        return float(np.clip(mix, 0.0, 1.0)), next_state, "TransitionNeural"

    def get_risk(self, task_context, history_item, prev_state=None, has_prior_error=False, is_first_step=False, agent_description="", step_idx=0, total_steps=1, history_context=""):
        state_full = self.state_manager.extract_state(
            task_context,
            history_item,
            has_prior_error,
            agent_description,
            history_context=history_context,
        )
        return 0.0, state_full, "DeprecatedSingleStep"

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
        "TransitionNeural": {"attempted": 0, "detected": 0, "correct": 0},
        "EpistemicFallback": {"attempted": 0, "detected": 0, "correct": 0},
        "DeprecatedSingleStep": {"attempted": 0, "detected": 0, "correct": 0},
        "No_Detection": {"attempted": 0, "detected": 0, "correct": 0}
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
        detected_level = None
        outcome = "MISSED"
        agent_match = False
        
        limit = min(len(history), mistake_step + 5)
        
        has_prior_error = False
        max_observed_risk = 0.0
        step_records = []
        
        for i in range(max(0, limit - 1)):
            risk, next_state, used_level = markov_model.get_transition_risk(
                task_context=task_context,
                history=history,
                step_idx=i,
                has_prior_error=has_prior_error,
                system_prompts=system_prompts,
            )

            target_idx = i + 1
            next_msg = history[target_idx]
            agent_name = next_msg.get('name', next_msg.get('role', 'Unknown'))
            risk = markov_model.calibrate_risk(risk, agent_name)
            step_records.append((target_idx, agent_name, risk))

            # Record attempt
            if used_level in level_stats:
                level_stats[used_level]["attempted"] += 1
            
            if target_idx == mistake_step:
                max_observed_risk = risk
            threshold = RISK_THRESHOLD
            
            # Only detect if not yet detected
            if target_idx < markov_model.min_detection_step:
                pass
            elif detected_at == -1 and risk > threshold:
                detected_at = target_idx
                detected_agent = agent_name
                detected_level = used_level
            
            _ = next_state
        
        if detected_at != -1:
            detected_agent = markov_model.predict_blame_agent(step_records, detected_at, window=2)
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
    for level_name in ["TransitionNeural", "EpistemicFallback", "DeprecatedSingleStep"]:
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
    datasets_dirs = [
        #"Who&When/Algorithm-Generated",，
        #"Who&When/Hand-Crafted",
        #"datasets/mmlu",
        "datasets/aqua",
        #"datasets/humaneval"
    ]
    
    all_files = []
    dataset_counts = {}
    
    for data_dir in datasets_dirs:
        if os.path.exists(data_dir):
            files = glob.glob(os.path.join(data_dir, "*.json"))
            # Skip empty files
            valid_files = [f for f in files if os.path.getsize(f) > 0]
            all_files.extend(valid_files)
            dataset_counts[data_dir] = len(valid_files)
            print(f"Loaded {len(valid_files)} files from {data_dir}")
        else:
            print(f"Warning: Directory {data_dir} not found")
    
    print(f"\nTotal files from all datasets: {len(all_files)}")
    print(f"Dataset breakdown: {dataset_counts}\n")
    
    # Shuffle combined dataset
    random.shuffle(all_files)
    
    split = int(len(all_files) * 0.5)
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
