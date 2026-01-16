import { useState, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { api } from '../services/api';
import type { ActionState, ActionCandidate, ActionButton, RouteFeedbackRequest } from '../services/api';

export interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  options?: ActionCandidate[];
  buttons?: ActionButton[];
  timestamp?: string;
}

// Context để tracking feedback
interface FeedbackContext {
  requestId: string;
  uiStrategy: string;
  topActions: ActionCandidate[];
}

export const useChatSession = () => {
  const [messages, setMessages] = useState<Message[]>([
    { 
      id: '1', 
      sender: 'bot', 
      text: 'Xin chào! Tôi có thể giúp gì cho bạn?',
      timestamp: new Date().toLocaleTimeString('vi-VN', { hour12: false }) + '.' + String(new Date().getMilliseconds()).padStart(3, '0')
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [session, setSession] = useState<ActionState | null>(null);

  const [lastUserText, setLastUserText] = useState<string>('');
  
  // Feedback tracking refs
  const feedbackContextRef = useRef<FeedbackContext | null>(null);
  const currentActionRef = useRef<{ actionId: string; requestId: string } | null>(null);

  const addMessage = (
    sender: 'user' | 'bot',
    text: string,
    options?: ActionCandidate[],
    buttons?: ActionButton[],
  ) => {
    const now = new Date();
    const timestamp = now.toLocaleTimeString('vi-VN', { hour12: false }) + '.' + String(now.getMilliseconds()).padStart(3, '0');
    setMessages(prev => [...prev, { id: uuidv4(), sender, text, options, buttons, timestamp }]);
  };

  // Send selection feedback to backend
  const sendSelectionFeedback = useCallback(async (
    actionId: string, 
    selectionIndex: number,
    selectionSource: 'click' | 'preselect'
  ) => {
    if (!feedbackContextRef.current) return;
    
    const { requestId, uiStrategy } = feedbackContextRef.current;
    
    const feedbackData: RouteFeedbackRequest = {
      request_id: requestId,
      user_id: 'user',
      session_id: session?.session_id,
      selected_action: actionId,
      selection_index: selectionIndex,
      selection_source: selectionSource,
      ui_strategy: uiStrategy,
    };

    try {
      await api.feedbackRoute(feedbackData);
      console.log('[Feedback] Selection sent:', feedbackData);
    } catch (error) {
      console.error('[Feedback] Failed to send selection:', error);
    }
  }, [session?.session_id]);

  // Send outcome feedback to backend
  const sendOutcomeFeedback = useCallback(async (
    status: 'confirmed' | 'canceled'
  ) => {
    if (!currentActionRef.current) return;
    
    const { actionId, requestId } = currentActionRef.current;
    
    try {
      await api.feedbackOutcome({
        request_id: requestId,
        user_id: 'user',
        session_id: session?.session_id,
        action_id: actionId,
        status: status,
      });
      console.log('[Feedback] Outcome sent:', { requestId, actionId, status });
    } catch (error) {
      console.error('[Feedback] Failed to send outcome:', error);
    }
  }, [session?.session_id]);

  const handleActionState = useCallback((state: ActionState) => {
    setSession(state);
    const msg = state.message || `Trạng thái: ${state.status}`;
    addMessage('bot', msg, undefined, state.buttons);

    // Send outcome feedback when action completes
    if (['COMMITTED', 'CONFIRMED'].includes(state.status)) {
      sendOutcomeFeedback('confirmed');
      setSession(null);
      setLastUserText('');
      currentActionRef.current = null;
      feedbackContextRef.current = null;
    } else if (state.status === 'CANCELED') {
      sendOutcomeFeedback('canceled');
      setSession(null);
      setLastUserText('');
      currentActionRef.current = null;
      feedbackContextRef.current = null;
    }
  }, [sendOutcomeFeedback]);

  const startNewAction = useCallback(async (
    actionId: string, 
    initialText: string = "",
    selectionIndex: number = 0,
    selectionSource: 'click' | 'preselect' = 'click'
  ) => {
    const sessionId = uuidv4();
    setLoading(true);
    
    // Send selection feedback
    await sendSelectionFeedback(actionId, selectionIndex, selectionSource);
    
    // Store current action for outcome tracking
    if (feedbackContextRef.current) {
      currentActionRef.current = {
        actionId,
        requestId: feedbackContextRef.current.requestId,
      };
    }
    
    const contextText = initialText || lastUserText;
    
    try {
      const state = await api.startAction(sessionId, actionId, contextText).then(r => r.data);
      handleActionState(state);
    } catch (e) {
      addMessage('bot', 'Không thể khởi tạo hành động.');
    } finally {
      setLoading(false);
    }
  }, [sendSelectionFeedback, lastUserText, handleActionState]);

  const sendMessage = useCallback(async (text: string) => {
    addMessage('user', text);
    setLoading(true);
    setLastUserText(text);

    try {
      if (session && !['COMMITTED', 'CANCELED', 'CONFIRMED'].includes(session.status)) {
        // In session
        const newState = await api.interactAction(session.session_id, text).then(r => r.data);
        handleActionState(newState);
      } else {
        // Routing
        const res = await api.route(text).then(r => r.data);
        
        // Store feedback context for later use
        feedbackContextRef.current = {
          requestId: res.request_id,
          uiStrategy: res.ui_strategy,
          topActions: res.top_actions,
        };

        if (res.ui_strategy === 'PRESELECT' && res.top_actions.length > 0) {
          addMessage('bot', res.message);
          // Auto-select first action with preselect source
          startNewAction(res.top_actions[0].action_id, text, 0, 'preselect');
        } else if (['TOP_3', 'CLARIFY'].includes(res.ui_strategy)) {
          addMessage('bot', res.message, res.top_actions);
        } else {
          addMessage('bot', res.message);
        }
      }
    } catch (error) {
      addMessage('bot', 'Đã có lỗi xảy ra. Vui lòng thử lại.');
      console.error(error);
    } finally {
      setLoading(false);
    }
  }, [session, handleActionState, startNewAction]);

  // Wrapper for startNewAction that finds selection index
  const handleActionSelect = useCallback((actionId: string) => {
    const index = feedbackContextRef.current?.topActions.findIndex(
      a => a.action_id === actionId
    ) ?? 0;
    startNewAction(actionId, '', index, 'click');
  }, [startNewAction]);

  return {
    messages,
    loading,
    session,
    sendMessage,
    startNewAction: handleActionSelect, // Use wrapper for UI clicks
  };
};
