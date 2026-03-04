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

# --- 2. Discrete Markov Chain ---

class DiscreteStateMarkov:
    """
    Hierarchical Markov Model for Risk Prediction.
    Levels:
    1. Full Transition: (Prev_Role, Prev_Action) -> (Curr_Role) [Most Precise]
    2. Role Transition: (Prev_Role) -> (Curr_Role) [Generalization]
    3. Base Role Risk: (Curr_Role) [Fallback]
    """
    def __init__(self, classifier):
        self.state_manager = PreDefinedStateManager(classifier)
        
        # Level 1: Full Transition (Prev_Role, Prev_Action, Prev_Error) -> Curr_Role
        self.trans_full_counts = defaultdict(Counter)
        self.trans_full_mistakes = defaultdict(Counter)
        
        # Level 2: Role Transition (Prev_Role) -> Curr_Role
        self.trans_role_counts = defaultdict(Counter)
        self.trans_role_mistakes = defaultdict(Counter)
        
        # Level 3: Base Role Risk P(Mistake | Curr_Role)
        self.base_counts = Counter()
        self.base_mistakes = Counter()
        
        # Risk Maps
        self.risk_map_full = {}
        self.risk_map_role_trans = {}
        self.risk_map_base = {}
        
        self.optimal_threshold = 0.1 

    def optimize_threshold(self, dataset):
        """"
        Find the best threshold that separates Safe steps from Mistake steps.
        """
        print("\nOptimizing Detection Threshold...")
        safe_scores = []
        mistake_scores = []
        
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
                
                risk, state = self.get_risk(
                    task_context, 
                    curr_msg, 
                    prev_state, 
                    has_prior_error, 
                    is_first_step=(i==0),
                    agent_description=desc
                )
                
                if i == mistake_step:
                    mistake_scores.append(risk)
                else:
                    safe_scores.append(risk)
                
                (_, _, action, _) = state
                if action == ActionType.FAIL: has_prior_error = True
                prev_state = state

        if not mistake_scores:
            print("Warning: No mistake steps found to optimize threshold.")
            return

        avg_safe = np.mean(safe_scores) if safe_scores else 0.0
        avg_mistake = np.mean(mistake_scores)
        std_safe = np.std(safe_scores) if safe_scores else 0.0
        
        # Strategy: The training set has exact transitions (Level 1) yielding high risk,
        # but the test set often relies on Level 2/3 fallback due to sparsity.
        # This causes a "Threshold Mismatch" - we must lower the threshold closer to Safe distribution.
        
        # We target a threshold that catches anomalies (3-sigma from safe mean)
        # But limited by the mistake mean to avoid absurdity.
        
        # Strategy: Use Percentiles instead of Means.
        # Goal: Recall > Precision. We want to catch the mistakes.
        # Set threshold to the 10th percentile of Mistake Scores in training.
        # This ensures we catch 90% of training mistakes, assuming distribution holds.
        
        target_T = np.percentile(mistake_scores, 10)
        
        # Safety check: Don't go below 10th percentile of Safe (too much noise)
        min_T = np.percentile(safe_scores, 10)
        target_T = max(target_T, min_T)
        
        self.optimal_threshold = float(target_T)
        
        print(f"Stats: Avg Safe Risk={avg_safe:.4f}, Avg Mistake Risk={avg_mistake:.4f}")
        print(f"Selected Optimal Threshold (10th %ile of Mistakes): {self.optimal_threshold:.4f}")
        
        print(f"Stats: Avg Safe Risk={avg_safe:.4f} (std={std_safe:.4f}), Avg Mistake Risk={avg_mistake:.4f}")
        print(f"Selected Optimal Threshold: {self.optimal_threshold:.4f}")

    def fit(self, dataset):
        print("Fitting Hierarchical Markov Model...")
        limit_files = len(dataset)
        
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
                except: continue
                
                (c_t, c_r, c_a, c_e) = curr_state
                # CRITICAL: Include Action in Target Key
                curr_target_key = (c_t, c_r, c_a)
                
                is_mistake = (i == mistake_step)
                
                # Update Level 3: Base Role+Action Stats
                self.base_counts[curr_target_key] += 1
                if is_mistake: self.base_mistakes[curr_target_key] += 1
                
                if prev_state_tuple:
                    (p_t, p_r, p_a, p_e) = prev_state_tuple
                    prev_role_key = (p_t, p_r)
                    
                    # Update Level 1: Full Transition -> Role+Action
                    self.trans_full_counts[prev_state_tuple][curr_target_key] += 1
                    if is_mistake: self.trans_full_mistakes[prev_state_tuple][curr_target_key] += 1
                    
                    # Update Level 2: Role Transition -> Role+Action
                    self.trans_role_counts[prev_role_key][curr_target_key] += 1
                    if is_mistake: self.trans_role_mistakes[prev_role_key][curr_target_key] += 1
                
                if c_a == ActionType.FAIL: has_prior_error = True
                prev_state_tuple = curr_state

        self._compute_risks()
        self.optimize_threshold(dataset)

    def _compute_risks(self):
        print("Computing Hierarchical Risks...")
        
        # Heuristic: Failures are rare events. We boost their signal significantly.
        # If even 1 mistake happened, it's a "Risky Zone".
        FAILURE_BOOST = 2.0
        
        # 1. Base Risks
        for k, n in self.base_counts.items():
            m = self.base_mistakes[k]
            self.risk_map_base[k] = (m * FAILURE_BOOST + 0.1) / (n + 1.0)
            
        # 2. Role Transition Risks
        for p_key, sub in self.trans_role_counts.items():
            for c_key, n in sub.items():
                m = self.trans_role_mistakes[p_key][c_key]
                self.risk_map_role_trans[(p_key, c_key)] = (m * FAILURE_BOOST + 0.2) / (n + 1.0)
                
        # 3. Full Transition Risks
        for p_state, sub in self.trans_full_counts.items():
            for c_key, n in sub.items():
                m = self.trans_full_mistakes[p_state][c_key]
                self.risk_map_full[(p_state, c_key)] = (m * FAILURE_BOOST + 0.5) / (n + 1.0)

    def get_risk(self, task_context, history_item, prev_state=None, has_prior_error=False, is_first_step=False, agent_description=""):
        state_full = self.state_manager.extract_state(task_context, history_item, has_prior_error, agent_description)
        (c_t, c_r, c_a, _) = state_full
        
        # New Target Key: (Role + Action)
        curr_target_key = (c_t, c_r, c_a)
        
        risks = []
        
        # Level 3: Base (Default)
        r_base = self.risk_map_base.get(curr_target_key, 0.15)
        risks.append(r_base)
        
        if prev_state:
            (p_t, p_r, _, _) = prev_state
            prev_role_key = (p_t, p_r)
            
            # Level 2: Role Transition -> Role+Action
            if (prev_role_key, curr_target_key) in self.risk_map_role_trans:
                r2 = self.risk_map_role_trans[(prev_role_key, curr_target_key)]
                risks.append(r2)
            else:
                # UNSEEN Transition!
                # Allow innocent until proven guilty (low default for novel transitions)
                # This reduces false positives on safe but new actions.
                risks.append(0.15)
                
            # Level 1: Full Transition -> Role+Action (Best Precision)
            if (prev_state, curr_target_key) in self.risk_map_full:
                r1 = self.risk_map_full[(prev_state, curr_target_key)]
                risks.append(r1)
        
        # PARANOID STRATEGY: Take the Weighted Max to capture any signal of danger
        # But we still weight specific (r1) more if it exists?
        # Actually, Max is robust against dilution.
        # But we dampen it slightly to avoid single-sample noise.
        
        final_risk = max(risks)
        
        return final_risk, state_full
            # Note: We don't penalize missing Level 1 as heavily as missing Level 2,
            # because "Full State" is naturally sparse. Level 2 miss is a stronger anomaly signal.

        return risk, state_full

        return risk, state_full

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
    print("\n--- Running Finite State Risk Evaluation ---")
    
    stats = {
        "total_cases": 0,
        "valid_cases": 0,
        "detected": 0,
        "early_detection": 0,
        "agent_hit": 0,
        "detection_with_correct_agent": 0
    }
    
    # We define a detection threshold based on the risk scores we observed during training
    # Dynamic threshold from training
    RISK_THRESHOLD = markov_model.optimal_threshold
    print(f"Using Optimal Risk Threshold: {RISK_THRESHOLD:.4f}")
    
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
        outcome = "MISSED"
        agent_match = False
        
        limit = min(len(history), mistake_step + 5) # Look a bit past to see if we catch it slightly late
        
        prev_state = None
        has_prior_error = False
        max_observed_risk = 0.0
        
        for i in range(limit):
            curr_msg = history[i]
            agent_name = curr_msg.get('name', curr_msg.get('role', 'Unknown'))
            agent_description = system_prompts.get(agent_name, "")
            
            # 1. Get Risk
            risk, state = markov_model.get_risk(
                task_context, 
                curr_msg, 
                prev_state=prev_state, 
                has_prior_error=has_prior_error, 
                is_first_step=(i==0),
                agent_description=agent_description
            )
            if i == mistake_step:
                max_observed_risk = risk
            
            # 2. Heuristic Detection Logic
            # If Risk is high, flag it.
            if risk > RISK_THRESHOLD:
                detected_at = i
                detected_agent = agent_name
                break
            
            # Update Context for NEXT step
            (_, _, action, _) = state
            if action == ActionType.FAIL:
                has_prior_error = True
            
            prev_state = state
        
        if detected_at != -1:
            # Check if detected agent matches the mistake agent
            agent_match = detected_agent == mistake_agent
            if agent_match:
                stats["agent_hit"] += 1
            
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
    # data_dir = "Who&When/Algorithm-Generated"
    data_dir = "Who&When/Hand-Crafted"
    all_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    # Shuffle
    random.shuffle(all_files)
    
    split = int(len(all_files) * 0.2)
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
