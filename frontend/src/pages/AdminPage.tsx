import React, { useState, useEffect } from 'react';
import { Tabs, Input, Button, message, Spin } from 'antd';
import { api } from '../services/api';
import { Save } from 'lucide-react';

const { TextArea } = Input;

const AdminPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [actionsYaml, setActionsYaml] = useState('');
  const [rulesYaml, setRulesYaml] = useState('');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const actionsRes = await api.getActions();
      setActionsYaml(actionsRes.data.content);

      const rulesRes = await api.getRules();
      setRulesYaml(rulesRes.data.content);
    } catch (error) {
      message.error('Không tải được cấu hình');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveActions = async () => {
    setLoading(true);
    try {
      await api.updateActions(actionsYaml);
      message.success('Đã lưu Action Catalog thành công');
    } catch (error) {
      message.error('Lỗi khi lưu Action Catalog. Kiểm tra cú pháp YAML.');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveRules = async () => {
    setLoading(true);
    try {
      await api.updateRules(rulesYaml);
      message.success('Đã lưu Keyword Rules thành công');
    } catch (error) {
      message.error('Lỗi khi lưu Keyword Rules. Kiểm tra cú pháp YAML.');
    } finally {
      setLoading(false);
    }
  };

  const items = [
    {
      key: '1',
      label: 'Action Catalog',
      children: (
        <div className="flex h-[72vh] flex-col gap-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-500">Chỉnh sửa config/action_catalog.yaml</span>
            <Button type="primary" icon={<Save size={16} />} onClick={handleSaveActions}>
              Lưu thay đổi
            </Button>
          </div>
          <TextArea
            value={actionsYaml}
            onChange={e => setActionsYaml(e.target.value)}
            className="flex-1 rounded-xl border border-gray-200 bg-white font-mono text-sm text-slate-900"
            style={{ resize: 'none' }}
          />
        </div>
      ),
    },
    {
      key: '2',
      label: 'Keyword Rules',
      children: (
        <div className="flex h-[72vh] flex-col gap-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-500">Chỉnh sửa config/keyword_rules.yaml</span>
            <Button type="primary" icon={<Save size={16} />} onClick={handleSaveRules}>
              Lưu thay đổi
            </Button>
          </div>
          <TextArea
            value={rulesYaml}
            onChange={e => setRulesYaml(e.target.value)}
            className="flex-1 rounded-xl border border-gray-200 bg-white font-mono text-sm text-slate-900"
            style={{ resize: 'none' }}
          />
        </div>
      ),
    },
  ];

  return (
    <Spin spinning={loading}>
      <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-xl shadow-slate-200">
        <Tabs
          defaultActiveKey="1"
          items={items}
          className="[&_.ant-tabs-tab-btn]:text-slate-700 [&_.ant-tabs-tab]:px-4 [&_.ant-tabs-tab]:py-2 [&_.ant-tabs-ink-bar]:bg-amber-400"
        />
      </div>
    </Spin>
  );
};

export default AdminPage;
