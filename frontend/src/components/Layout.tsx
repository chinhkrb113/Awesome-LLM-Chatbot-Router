import React from 'react';
import { Layout as AntLayout } from 'antd';
import { Link, useLocation } from 'react-router-dom';
import { MessageSquare, Settings, Sparkles } from 'lucide-react';

const { Header, Content, Footer } = AntLayout;

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();

  const items = [
    {
      key: '/',
      icon: <MessageSquare size={16} />,
      label: 'Chat Interface',
    },
    {
      key: '/admin',
      icon: <Settings size={16} />,
      label: 'Admin Dashboard',
    },
  ];

  return (
    <AntLayout className="relative min-h-screen overflow-hidden bg-[#f6f7fb] text-slate-900">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-24 left-10 h-72 w-72 rounded-full bg-white blur-3xl" />
        <div className="absolute top-0 right-16 h-80 w-80 rounded-full bg-amber-100/70 blur-3xl" />
        <div className="absolute bottom-[-160px] left-1/3 h-80 w-80 rounded-full bg-cyan-100/60 blur-3xl" />
      </div>

      <Header className="sticky top-0 z-20 border-b border-gray-200/80 bg-white/85 px-4 py-3 backdrop-blur-lg shadow">
        <div className="mx-auto flex max-w-6xl items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-200 via-white to-cyan-300 text-slate-900 shadow">
              <Sparkles size={18} />
            </div>
            <div>
              <div className="text-lg font-semibold tracking-tight text-slate-900">Hybrid Router Bot</div>
              <p className="text-xs text-slate-500">Lux-grade conversational control</p>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2 rounded-full border border-gray-200 bg-white px-1.5 py-1 shadow">
            {items.map(item => {
              const active = location.pathname === item.key;
              return (
                <Link
                  key={item.key}
                  to={item.key}
                  className={`flex items-center gap-2 rounded-full px-3 py-2 text-sm transition-all duration-200 ${
                    active
                      ? 'bg-slate-900 text-white shadow'
                      : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </Header>

      <Content className="relative px-4 pb-12 sm:px-6">
        <div className="mx-auto max-w-6xl pt-24 sm:pt-24">{children}</div>
      </Content>

      <Footer className="border-t border-gray-200 bg-white/80 text-center text-slate-500 backdrop-blur">
        Hybrid Intent Router System ©2026
      </Footer>
    </AntLayout>
  );
};

export default Layout;
