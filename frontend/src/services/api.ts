import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ActionCandidate {
  action_id: string;
  friendly_name?: string;
  rule_score: number;
  embed_score: number;
  final_score: number;
  reasoning: string[];
}

export interface RouterOutput {
  request_id: string;  // Added: để tracking feedback
  top_actions: ActionCandidate[];
  ui_strategy: 'PRESELECT' | 'TOP_3' | 'CLARIFY';
  message: string;
}

// Feedback request types
export interface RouteFeedbackRequest {
  request_id: string;
  user_id?: string;
  session_id?: string;
  selected_action?: string;
  selection_index?: number;
  selection_source?: string;
  ui_strategy?: string;
}

export interface OutcomeFeedbackRequest {
  request_id: string;
  user_id?: string;
  session_id?: string;
  action_id?: string;
  status: 'confirmed' | 'canceled';
}

export interface SlotValue {
    name: string;
    value: any;
    confidence?: number;
    source?: string;
}

export interface ActionButton {
    label: string;
    value: string;
    style: 'primary' | 'default' | 'danger';
}

export interface ActionState {
    session_id: string;
    action_id: string;
    status: 'INIT' | 'COLLECTING' | 'DRAFT' | 'CONFIRMED' | 'COMMITTED' | 'CANCELED';
    slots: Record<string, SlotValue>;
    history: string[];
    message?: string;
    buttons?: ActionButton[];
}

export const api = {
    route: (text: string, userId: string = 'user', requestId?: string) => 
        axios.post<RouterOutput>(`${API_URL}/route`, { 
            text, 
            user_id: userId,
            request_id: requestId 
        }),
    
    startAction: (sessionId: string, actionId: string, initialText: string = "") =>
        axios.post<ActionState>(`${API_URL}/action/start`, { 
            session_id: sessionId, 
            action_id: actionId,
            initial_text: initialText 
        }),
        
    interactAction: (sessionId: string, text: string) =>
        axios.post<ActionState>(`${API_URL}/action/interact`, { session_id: sessionId, text }),

    // Feedback APIs for Learning Loop
    feedbackRoute: (data: RouteFeedbackRequest) =>
        axios.post(`${API_URL}/feedback/route`, data),

    feedbackOutcome: (data: OutcomeFeedbackRequest) =>
        axios.post(`${API_URL}/feedback/outcome`, data),

    // Admin APIs
    getActions: () => axios.get<{content: string}>(`${API_URL}/admin/config/actions`),
    updateActions: (content: string) => axios.post(`${API_URL}/admin/config/actions`, { content }),

    getRules: () => axios.get<{content: string}>(`${API_URL}/admin/config/rules`),
    updateRules: (content: string) => axios.post(`${API_URL}/admin/config/rules`, { content }),
};
