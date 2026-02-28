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
    CODING_AGENT = "CODING_AGENT"       # Python_Expert, Excel_Expert, DataAnalysis_Expert...
    VERIFIER_AGENT = "VERIFIER_AGENT"   # Verification_Expert, Validation_Expert...
    SEARCH_AGENT = "SEARCH_AGENT"       # General Search
    WEB_SURFER = "WEB_SURFER"           # WebSurfer (browsing, clicking, typing)
    FILE_SURFER = "FILE_SURFER"         # FileSurfer (local file ops)
    ORCHESTRATOR = "ORCHESTRATOR"       # Orchestrator (planning, delegating)
    ASSISTANT = "ASSISTANT"             # Assistant (general purpose)
    TERMINAL = "TERMINAL"               # Computer_terminal
    MANAGER_AGENT = "MANAGER_AGENT"     # General Manager (fallback)
    OTHER = "OTHER"

    @classmethod
    def all(cls):
        return [
            cls.CODING_AGENT, cls.VERIFIER_AGENT, cls.SEARCH_AGENT, 
            cls.WEB_SURFER, cls.FILE_SURFER, cls.ORCHESTRATOR, cls.ASSISTANT,
            cls.TERMINAL, cls.MANAGER_AGENT, cls.OTHER
        ]

class ActionType:
    WRITE_CODE = "WRITE_CODE"
    EXEC_SUCCESS = "EXEC_SUCCESS"
    EXEC_FAIL = "EXEC_FAIL"
    EXECUTION_OUTPUT = "EXECUTION_OUTPUT" # Output from terminal
    SEARCH = "SEARCH"
    BROWSE = "BROWSE"           # Web interaction (click, scroll, type)
    READ_FILE = "READ_FILE"     # Reading files
    WRITE_FILE = "WRITE_FILE"   # Writing files
    VERIFY = "VERIFY"
    PLAN = "PLAN"
    THOUGHT = "THOUGHT"         # Internal reasoning / Ledger updates
    DELEGATE = "DELEGATE"       # Assigning tasks to other agents
    TERMINATE = "TERMINATE"
    CHAT = "CHAT" # General conversation
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def all(cls):
        return [
            cls.WRITE_CODE, cls.EXEC_SUCCESS, cls.EXEC_FAIL, 
            cls.SEARCH, cls.BROWSE, cls.READ_FILE, cls.WRITE_FILE,
            cls.VERIFY, cls.PLAN, cls.THOUGHT, cls.DELEGATE,
            cls.TERMINATE, cls.CHAT, cls.UNKNOWN
        ]

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
            device_map=self.device
        )
        # Fix for "Setting `pad_token_id` to `eos_token_id`" warning
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def classify_state(self, task_context, history_item):
        """
        Uses LLM to classify Task, Role, and Action.
        """
        content = history_item.get('content', '')[:1000] # Truncate for speed
        agent_name = history_item.get('name', history_item.get('role', 'Unknown'))
        
        # 1. System Prompt
        system_prompt = (
            "You are a specialized classifier for Multi-Agent Systems. "
            "Your job is to categorize the Agent's Role and the Action Type into predefined categories.\n"
            f"Allowed ROLES: {', '.join(AgentRole.all())}\n"
            f"Allowed ACTIONS: {', '.join(ActionType.all())}\n"
            f"Allowed TASK TYPES: {', '.join(TaskType.all())}\n"
            "Output JSON format only: {\"role\": \"...\", \"action\": \"...\", \"task\": \"...\"}"
        )
        
        # 2. User Prompt
        user_prompt = (
            f"Task: {task_context[:200]}\n"
            f"Agent Name: {agent_name}\n"
            f"Message Content: {content}\n\n"
            "Classify."
        )
        
        # Simple prompt construction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Ensure pad_token_id is set
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                max_new_tokens=60,
                do_sample=False, 
                temperature=None,
                top_p=None
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON
        try:
            # Simple fuzzy extraction
            json_str = generated_text.replace("```json", "").replace("```", "").strip()
            # If model chats, try to find the dict
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(json_str[start:end])
            else:
                data = {}
        except:
            data = {}
            
        # Fallback to defaults (using simple heuristics just in case LLM fails)
        task = data.get("task", TaskType.OTHER)
        role = data.get("role", AgentRole.OTHER)
        action = data.get("action", ActionType.UNKNOWN)
        
        # Validate against allowed lists
        if task not in TaskType.all(): task = TaskType.OTHER
        if role not in AgentRole.all(): role = AgentRole.OTHER
        if action not in ActionType.all(): action = ActionType.CHAT
        
        return task, role, action

class PreDefinedStateManager:
    """
    Wrapper that now uses LLM Classifier instead of if-else.
    Includes caching to avoid re-running LLM on same identical messages.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.cache = {}
    
    def extract_state(self, task_context, history_item, has_prior_error=False):
        # Create a cache key from content hash + agent
        content_hash = hash(history_item.get('content', ''))
        key = (task_context[:50], history_item.get('name'), content_hash)
        
        if key in self.cache:
            task_type, role, action = self.cache[key]
        else:
            task_type, role, action = self.classifier.classify_state(task_context, history_item)
            self.cache[key] = (task_type, role, action)
            
        return (task_type, role, action, has_prior_error)

# --- 2. Discrete Markov Chain ---

class DiscreteStateMarkov:
    """
    Builds transition probabilities between Defined States.
    P(Next_State | Current_State)
    Also tracks: P(Mistake | Current_State) -> "State Riskiness"
    """
    def __init__(self, classifier):
        self.state_manager = PreDefinedStateManager(classifier)
        
        # Transitions: state -> next_state -> count
        self.transitions = defaultdict(Counter)
        self.transition_mistake_counts = defaultdict(Counter) # (prev, curr) -> count of mistakes
        
        # Mistake Counts: state -> count (how often this state IS the mistake step)
        self.state_counts = Counter()
        
        # Start State Stats (Step 0)
        self.start_role_counts = Counter()
        self.start_role_success = Counter()
        self.start_role_failure = Counter()
        
        # Contrastive Stats
        self.success_state_counts = Counter() # Times state appears in SUCCESSFUL (or non-mistake) flows
        self.failure_state_counts = Counter() # Times state appears in FAILED flows (specifically around mistake)
        
        # Pre-Action Risk Scores (Role-based, ignoring Action)
        self.role_risk_scores = {} 
        self.transition_role_risk_scores = {}
        self.start_role_risk_scores = {} # Specific for step 0: (Task, Role) -> Risk
        
        # Threshold for detection
        self.optimal_threshold = 0.1 # Default fallback

    def optimize_threshold(self, dataset):
        """"
        Find the best threshold that separates Safe steps from Mistake steps.
        """
        print("\nOptimizing Detection Threshold...")
        safe_scores = []
        mistake_scores = []
        
        limit_files = len(dataset)
        # Use a subset of training data to save time if needed, but here we use all
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            
            prev_state = None
            has_prior_error = False
            
            for i in range(len(history)):
                curr_msg = history[i]
                risk, state = self.get_risk(task_context, curr_msg, prev_state, has_prior_error, is_first_step=(i==0))
                
                if i == mistake_step:
                    mistake_scores.append(risk)
                else:
                    safe_scores.append(risk)
                
                # Update context
                (_, _, action, _) = state
                if action == ActionType.EXEC_FAIL: has_prior_error = True
                prev_state = state

        if not mistake_scores:
            print("Warning: No mistake steps found to optimize threshold.")
            return

        avg_safe = np.mean(safe_scores) if safe_scores else 0.0
        avg_mistake = np.mean(mistake_scores)
        
        # Simple separation logic: Average of the means
        
        # Strategy 3: Weighted closer to mistake to avoid noise, but ensure we catch peaks
        self.optimal_threshold = (avg_safe * 0.3 + avg_mistake * 0.7)
        
        print(f"Stats: Avg Safe Risk={avg_safe:.4f}, Avg Mistake Risk={avg_mistake:.4f}")
        print(f"Selected Optimal Threshold: {self.optimal_threshold:.4f}")

    def fit(self, dataset):
        print("Fitting Discrete Markov Chain on Finite predefined states (using LLM Classification)...")
        
        limit_files = len(dataset)
        
        for idx in tqdm(range(limit_files)):
            json_data = dataset[idx]
            history = json_data.get('history', [])
            ms = json_data.get('mistake_step', -1)
            mistake_step = int(ms) if (ms is not None and str(ms).isdigit()) else -1
            task_context = json_data.get('question', '')
            
            prev_state_tuple = None
            has_prior_error = False
            
            for i in range(len(history)):
                curr_msg = history[i]
                
                # Use LLM Classifier via state_manager
                try:
                    curr_state_tuple = self.state_manager.extract_state(task_context, curr_msg, has_prior_error)
                except Exception as e:
                    print(f"Error extracting state: {e}")
                    continue

                # Update Counts
                self.state_counts[curr_state_tuple] += 1
                
                # SPECIAL: Track Step 0 separate from general pool
                if i == 0:
                    (t, r, a, e) = curr_state_tuple
                    self.start_role_counts[(t, r)] += 1
                    if mistake_step == 0:
                         self.start_role_failure[(t, r)] += 2 # Boost mistake weight
                    else:
                         self.start_role_success[(t, r)] += 1

                # Contrastive Logic
                if mistake_step != -1:
                    # In a FAILURE trajectory
                    dist_to_fail = i - mistake_step
                    # If this state is exactly the mistake or very close to it (e.g. -1, 0)
                    if -2 <= dist_to_fail <= 0:
                        self.failure_state_counts[curr_state_tuple] += 2 # Weighted higher
                    else:
                        # It appeared in a failed trace but wasn't the cause -> Slight penalty or neutral
                        self.success_state_counts[curr_state_tuple] += 0.5 
                else:
                    # Pure SUCCESS trajectory
                    self.success_state_counts[curr_state_tuple] += 1

                if prev_state_tuple:
                    self.transitions[prev_state_tuple][curr_state_tuple] += 1
                
                # Mistake Tracking
                if i == mistake_step:
                    if prev_state_tuple:
                        self.transition_mistake_counts[prev_state_tuple][curr_state_tuple] += 1
                
                # Update Context for NEXT step
                (_, _, action, _) = curr_state_tuple
                if action == ActionType.EXEC_FAIL:
                    has_prior_error = True
                
                prev_state_tuple = curr_state_tuple
        
        self._compute_risks()
        
        # Optimize threshold using training data
        self.optimize_threshold(dataset)
        
    def _compute_risks(self):
        """
        Calculates Risk using Contrastive Odds Ratio.
        Risk(S) = (Count_Fail(S) + alpha) / (Count_Success(S) + Count_Fail(S) + alpha + beta)
        But specifically boosting states unique to failures.
        """
        print("\nComputing State & Transition Risks (Contrastive)...")
        
        # 3. Pre-Action Risk (Aggregating over Actions)
        # We aggregate counts for (Task, Role, *, PriorError)
        role_success = Counter()
        role_failure = Counter()
        role_counts = Counter()
        
        for state, count in self.state_counts.items():
            (t, r, a, e) = state
            role_key = (t, r, e)
            role_success[role_key] += self.success_state_counts[state]
            role_failure[role_key] += self.failure_state_counts[state]
            role_counts[role_key] += count
            
        for key, total in role_counts.items():
            fail_w = role_failure[key]
            succ_w = role_success[key]
            risk = (fail_w + 1) / (fail_w + succ_w + 10)
            if succ_w > 5 * fail_w: risk *= 0.5
            self.role_risk_scores[key] = risk

        # 4. Pre-Action Transition Risk
        # (PrevState) -> (Task, Role, *, PriorError)
        trans_role_mistakes = defaultdict(Counter)
        trans_role_counts = defaultdict(Counter)
        
        for prev_state in self.transitions:
            for curr_state, count in self.transitions[prev_state].items():
                (t, r, a, e) = curr_state
                role_key = (t, r, e)
                m_count = self.transition_mistake_counts[prev_state][curr_state]
                
                trans_role_mistakes[prev_state][role_key] += m_count
                trans_role_counts[prev_state][role_key] += count
                
        for prev_state, next_roles in trans_role_counts.items():
            for role_key, count in next_roles.items():
                m_count = trans_role_mistakes[prev_state][role_key]
                risk = (m_count * 2) / (count + 5)
                self.transition_role_risk_scores[(prev_state, role_key)] = risk

        # 5. Start State Risk (Step 0 Specific)
        for role_key, count in self.start_role_counts.items():
            fail_w = self.start_role_failure[role_key]
            succ_w = self.start_role_success[role_key]

            # Contrastive Learning Formula (Bayesian Smoothed Odds Ratio)
            risk = (fail_w + 0.5) / (fail_w + succ_w + 5.0)
            
            # Penalize highly successful starts
            if succ_w > 5 * fail_w:
                risk *= 0.5
            
            self.start_role_risk_scores[role_key] = risk


    def get_risk(self, task_context, history_item, prev_state=None, has_prior_error=False, is_first_step=False):
        """
        Returns combined risk using only Pre-Action information (Task & Role).
        Strictly ignores ActionType to prevent information leakage.
        """
        # We classify to get the Role, but we MUST ignore the Action for risk lookup
        state = self.state_manager.extract_state(task_context, history_item, has_prior_error)
        (task_type, role, action, _) = state

        # Construct the Pre-Action Key: (Task, Role, PriorError)
        role_key = (task_type, role, has_prior_error)
        
        # KEY CHANGE: For Step 0, we look at P(Mistake | Task, Role) AT START
        start_key = (task_type, role)

        if is_first_step:
            # 1. Use specific Start State Risk if available
            if start_key in self.start_role_risk_scores:
                return self.start_role_risk_scores[start_key], state
            
            # Fallback if specific start condition unseen: Use global role risk (conservative)
            base_risk = self.role_risk_scores.get(role_key, 0.0)
            return base_risk * 0.5, state 
        
        # for non-first steps:
        # Pre-Action Risk = max(Global Role Risk, Transition Role Risk)
        
        # 1. Base Pre-Action Risk (Global risk of this Role in this Task)
        base_risk = self.role_risk_scores.get(role_key, 0.0)
        
        # 2. Transition Pre-Action Risk (Risk of going from PrevState -> P(Mistake | PresentRole))
        trans_risk = 0.0
        if prev_state:
            trans_risk = self.transition_role_risk_scores.get((prev_state, role_key), 0.0)
        
        # Combine
        total_risk = max(base_risk, trans_risk)
        
        return total_risk, state

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
        "early_detection": 0
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
        
        if mistake_step == -1: continue
        
        stats["total_cases"] += 1
        stats["valid_cases"] += 1
        
        detected_at = -1
        outcome = "MISSED"
        
        limit = min(len(history), mistake_step + 5) # Look a bit past to see if we catch it slightly late
        
        prev_state = None
        has_prior_error = False
        
        for i in range(limit):
            curr_msg = history[i]
            
            # 1. Get Risk
            risk, state = markov_model.get_risk(task_context, curr_msg, prev_state=prev_state, has_prior_error=has_prior_error, is_first_step=(i==0))
            
            # 2. Heuristic Detection Logic
            # If Risk is high, flag it.
            if risk > RISK_THRESHOLD:
                detected_at = i
                break
            
            # Update Context for NEXT step
            (_, _, action, _) = state
            if action == ActionType.EXEC_FAIL:
                has_prior_error = True
            
            prev_state = state
        
        if detected_at != -1:
            if detected_at == mistake_step:
                stats["detected"] += 1
                outcome = "EXACT_HIT"
            elif detected_at < mistake_step:
                stats["early_detection"] += 1
                outcome = "EARLY_WARN"
            else:
                outcome = "LATE_MATCH"
            
            detect_ratios.append(detected_at / len(history))
            print(f"[ID {idx}] {outcome} @ {detected_at} (Tgt: {mistake_step}) - State: {state}")
        else:
            detect_ratios.append(1.0)
            print(f"[ID {idx}] MISSED {mistake_step}")

    print("\n=== Final Results ===")
    print(f"Total Valid Cases: {stats['valid_cases']}")
    print(f"Exact Hit: {stats['detected']} ({stats['detected']/stats['valid_cases']:.2%})")
    print(f"Early Warning: {stats['early_detection']} ({stats['early_detection']/stats['valid_cases']:.2%})")
    combined = stats['detected'] + stats['early_detection']
    print(f"Combined Recall (Exact+Early): {combined/stats['valid_cases']:.2%}")
    print(f"Avg Detection position ratio: {np.mean(detect_ratios):.4f}")

if __name__ == "__main__":
    data_dir = "Who&When/Algorithm-Generated"
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
