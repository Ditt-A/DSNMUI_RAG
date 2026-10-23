import { useState } from 'react';
import { useAmanaChat } from '@/hooks/use-amana-chat';
import { AmanaHeader } from '@/components/amana-header';
import { AmanaSidebar } from '@/components/amana-sidebar';
import { AmanaChat } from '@/components/amana-chat';
import { AmanaFilters } from '@/components/amana-filters';
import { AmanaInput } from '@/components/amana-input';
import { Toaster } from '@/components/ui/sonner';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { messages, isLoading, filters, updateFilter, sendMessage, clearChat } = useAmanaChat();

  return (
    <div className="flex h-screen w-full bg-slate-50/30 overflow-hidden font-sans">
      <AmanaSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={() => {
            clearChat();
            setSidebarOpen(false);
        }}
      />

      <div className="flex flex-col flex-1 min-w-0 h-full relative">
        <AmanaHeader onMenuClick={() => setSidebarOpen(true)} />

        <main className="flex-1 flex flex-col relative overflow-hidden">
          <AmanaChat messages={messages} isLoading={isLoading} />

          <div className="w-full px-4 md:px-8 pb-4 pt-2 bg-gradient-to-t from-background via-background/95 to-transparent z-10">
            <div className="max-w-4xl mx-auto w-full flex flex-col gap-2">
                <AmanaFilters filters={filters} onChange={updateFilter} disabled={isLoading} />
                <AmanaInput onSend={sendMessage} disabled={isLoading} />
                <div className="text-center mt-2">
                    <span className="text-[10px] text-muted-foreground/60">
                        AMANA dapat membuat kesalahan. Harap periksa kembali informasi penting referensi DSN-MUI.
                    </span>
                </div>
            </div>
          </div>
        </main>
      </div>

      <Toaster position="top-right" />
    </div>
  );
}
