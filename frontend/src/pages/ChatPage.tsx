import React, { useState, useEffect, useRef } from 'react';
import { Input, Button } from 'antd';
import { Send, Bot, User, Sparkles, ShieldCheck } from 'lucide-react';
import type { ActionState } from '../services/api';
import { useChatSession } from '../hooks/useChatSession';
import TypingIndicator from '../components/TypingIndicator';

const ChatPage: React.FC = () => {
  const { messages, loading, session, sendMessage, startNewAction } = useChatSession();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<any>(null); // antd Input ref type is complex, using any for simplicity or InputRef if imported

  const scrollToBottom = () => {
    // Chỉ scroll container tin nhắn, không scroll window
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  };

  // Auto focus input when loading finishes (bot replied)
  useEffect(() => {
    if (!loading) {
      inputRef.current?.focus();
    }
  }, [loading]);

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e?: React.SyntheticEvent | React.KeyboardEvent<HTMLInputElement>) => {
    e?.preventDefault();
    if (!inputValue.trim()) return;
    const text = inputValue;
    setInputValue('');
    await sendMessage(text);
  };

  const statusLabel = session ? `Session: ${stateToLabel(session.status)}` : 'Routing mode';

  return (
    <div className="relative space-y-6 text-slate-900">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -left-32 top-10 h-72 w-72 rounded-full bg-white blur-3xl" />
        <div className="absolute bottom-0 right-0 h-80 w-80 rounded-full bg-amber-100/60 blur-3xl" />
      </div>

      <section className="overflow-hidden rounded-3xl border border-gray-200 bg-white shadow-2xl shadow-slate-200/60">
        <div className="flex flex-wrap items-start justify-between gap-4 border-b border-gray-100 px-6 py-5">
          <div className="flex items-start gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-200 via-white to-cyan-300 text-slate-900 shadow-md">
              <Bot size={20} />
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-[0.25em] text-amber-600">Concierge</p>
              <h1 className="text-2xl font-semibold text-slate-900">Conversation Studio</h1>
              <p className="text-sm text-slate-600">Chạm nhẹ để định tuyến ý định với giao diện mượt và sang.</p>
            </div>
          </div>
          <div className="flex items-center gap-3 rounded-2xl border border-amber-100 bg-amber-50 px-4 py-3 text-xs text-amber-900 shadow">
            <span className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${loading ? 'bg-amber-400 animate-pulse' : 'bg-emerald-500'}`}
              />
              {loading ? 'Đang xử lý với độ mượt' : 'Sẵn sàng nhận câu hỏi'}
            </span>
            <span className="hidden sm:flex items-center gap-2 rounded-full bg-white px-3 py-1 text-amber-800 shadow-inner">
              <ShieldCheck size={14} />
              {statusLabel}
            </span>
          </div>
        </div>

        <div className="grid gap-6 px-6 pb-6 pt-5 lg:grid-cols-[1fr,18rem]">
          <div className="flex flex-col">
            <div className="relative flex-1 overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-b from-white via-slate-50 to-slate-100 shadow-inner shadow-slate-200">
              <div className="absolute inset-0 bg-gradient-to-b from-white/40 via-transparent to-white/70" />
              <div 
                ref={containerRef}
                className="relative flex flex-col gap-4 overflow-y-auto px-4 py-5 sm:px-6 sm:py-6 max-h-[60vh] min-h-[50vh] sm:max-h-[65vh]"
              >
                {messages.map(item => {
                  const isUser = item.sender === 'user';
                  return (
                    <div key={item.id} className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
                      <div
                        className={`mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                          isUser
                            ? 'bg-gradient-to-br from-amber-200 to-amber-400 text-slate-900 shadow-lg shadow-amber-400/50'
                            : 'border border-gray-200 bg-white text-amber-700 shadow'
                        }`}
                      >
                        {isUser ? <User size={16} /> : <Bot size={16} />}
                      </div>
                      <div className={`flex flex-col gap-1 max-w-[78%] ${isUser ? 'items-end' : 'items-start'}`}>
                        <div
                          className={`group w-full rounded-2xl border px-4 py-3 shadow-lg transition-all duration-300 ${
                            isUser
                              ? 'border-amber-200 bg-gradient-to-br from-amber-50 via-amber-100 to-amber-200 text-slate-900 shadow-amber-200/60'
                              : 'border-gray-200 bg-white text-slate-900 shadow'
                          }`}
                        >
                          <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.18em] text-slate-500">
                            <span>{isUser ? 'You' : 'Hybrid Bot'}</span>
                            <span className="text-[10px] text-slate-500">{isUser ? 'Live' : 'Guided'}</span>
                          </div>
                          <div
                            className={`mt-2 whitespace-pre-wrap leading-relaxed text-[15px] ${
                              isUser ? 'text-slate-900 font-semibold' : 'text-slate-800'
                            }`}
                          >
                            {item.text}
                          </div>
                          {item.options && (
                            <div className="mt-4 flex flex-col gap-3">
                              {item.options.map(opt => (
                                <button
                                  key={opt.action_id}
                                  type="button"
                                  onClick={() => startNewAction(opt.action_id)}
                                  className="group flex min-h-[48px] w-full items-center justify-between rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:border-amber-400 hover:bg-amber-50 hover:text-amber-900 hover:shadow-md active:translate-y-0 active:shadow-sm"
                                >
                                  <div className="flex items-center gap-3">
                                    <span className="text-left">{opt.friendly_name || opt.action_id}</span>
                                  </div>
                                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-bold text-slate-500 transition-colors group-hover:bg-amber-200/60 group-hover:text-amber-800">
                                    {Math.round(opt.final_score * 100)}%
                                  </span>
                                </button>
                              ))}
                            </div>
                          )}
                          {item.buttons && (
                            <div className="mt-3 flex flex-wrap gap-2">
                              {item.buttons.map((btn, idx) => {
                                const styleMap: Record<string, string> = {
                                  primary: 'bg-gradient-to-r from-amber-200 via-amber-300 to-amber-400 text-slate-900',
                                  danger: 'bg-gradient-to-r from-rose-400 to-amber-400 text-white',
                                  default: 'bg-slate-100 text-slate-900',
                                };
                                const baseClasses =
                                  'rounded-xl border px-3 py-2 text-xs font-semibold transition duration-200 shadow-md hover:-translate-y-0.5';
                                const styleClass =
                                  styleMap[btn.style] || 'bg-slate-100 text-slate-900';
                                const borderClass =
                                  btn.style === 'primary'
                                    ? 'border-amber-200'
                                    : btn.style === 'danger'
                                      ? 'border-rose-200/60'
                                      : 'border-gray-200 hover:border-amber-300';
                                return (
                                  <button
                                    key={`${btn.label}-${idx}`}
                                    type="button"
                                    onClick={() => sendMessage(btn.value)}
                                    className={`${baseClasses} ${styleClass} ${borderClass}`}
                                  >
                                    {btn.label}
                                  </button>
                                );
                              })}
                            </div>
                          )}
                        </div>
                        {item.timestamp && (
                          <span className="text-[10px] text-gray-400 px-1 font-mono">
                            {item.timestamp}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
                {loading && (
                  <div className="flex gap-3">
                    <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-gray-200 bg-white text-amber-700 shadow">
                      <Bot size={16} />
                    </div>
                    <TypingIndicator />
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>

            <form
              onSubmit={handleSubmit}
              className="mt-4 space-y-3 rounded-2xl border border-gray-200 bg-white p-4 shadow-xl shadow-slate-200"
            >
              <div className="flex flex-col gap-3 md:flex-row">
                <Input
                  ref={inputRef}
                  value={inputValue}
                  onChange={e => setInputValue(e.target.value)}
                  onPressEnter={handleSubmit}
                  placeholder="Nhập tin nhắn, hoặc chọn một gợi ý hành động..."
                  disabled={loading}
                  className="!h-12 !flex-1 !rounded-xl !border-slate-200 !bg-white !text-slate-900 !shadow-inner !shadow-slate-200 !placeholder:text-slate-500"
                />
                <Button
                  htmlType="submit"
                  type="primary"
                  icon={<Send size={16} />}
                  loading={loading}
                  className="!h-12 !rounded-xl !border-0 !bg-gradient-to-r !from-amber-200 !via-amber-300 !to-cyan-200 !px-5 !text-sm !font-semibold !text-slate-900 !shadow !shadow-amber-400/40 hover:!shadow-amber-400/60"
                >
                  Gửi
                </Button>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Sparkles size={14} className="text-amber-400" />
                <span>Enter để gửi nhanh. Các nút gợi ý hiển thị mượt mà.</span>
              </div>
            </form>
          </div>

          <aside className="space-y-3">
            <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-lg shadow-slate-200">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-amber-500">
                <ShieldCheck size={14} />
                <span>Trạng thái</span>
              </div>
              <p className="mt-2 text-lg font-semibold text-slate-900">{statusLabel}</p>
              <p className="mt-2 text-sm text-slate-600">
                Tương tác sẽ tự động chuyển giữa PRESELECT, TOP_3, CLARIFY và hành động.
              </p>
            </div>
            <div className="rounded-2xl border border-gray-200 bg-gradient-to-br from-white via-amber-50 to-cyan-50 p-4 shadow-lg shadow-slate-200">
              <p className="text-xs uppercase tracking-[0.18em] text-amber-500">Moodboard</p>
              <p className="mt-2 text-sm text-slate-700">
                Luxury + Soft Light: nền trắng/xám nhã, viền kính mỏng, vàng ngọc lam làm điểm nhấn.
              </p>
              <div className="mt-3 flex gap-2 text-[11px] font-semibold uppercase text-slate-900">
                <span className="rounded-full bg-amber-200 px-3 py-1 shadow-inner shadow-amber-500/40">Smooth</span>
                <span className="rounded-full bg-cyan-200 px-3 py-1 text-slate-900 shadow-inner shadow-cyan-500/30">
                  Modern
                </span>
              </div>
            </div>
          </aside>
        </div>
      </section>
    </div>
  );
};

const stateToLabel = (status: ActionState['status']) => {
  switch (status) {
    case 'INIT':
      return 'Khởi tạo';
    case 'COLLECTING':
      return 'Đang thu thập thông tin';
    case 'DRAFT':
      return 'Bản nháp';
    case 'CONFIRMED':
      return 'Đã xác nhận';
    case 'COMMITTED':
      return 'Đã chốt hành động';
    case 'CANCELED':
      return 'Đã hủy';
    default:
      return status;
  }
};

export default ChatPage;
