import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage } from '@/types/rag';
import { cn } from '@/lib/utils';
import { LoadingDots } from './amana-chat'; // Will extract this
import { AmanaSources } from './amana-sources';
import { Clock, ShieldCheck, Zap } from 'lucide-react';

interface AmanaMessageProps {
  message: ChatMessage;
}

export function AmanaMessage({ message }: AmanaMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn("flex w-full py-4", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] md:max-w-[75%] rounded-2xl px-5 py-4",
          isUser
            ? "bg-primary/10 text-foreground ml-auto rounded-tr-sm"
            : "bg-white border text-foreground mr-auto rounded-tl-sm shadow-[0_2px_8px_rgba(0,0,0,0.02)]"
        )}
      >
        <div className="flex items-center gap-2 mb-2">
            {!isUser && (
                <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                    <span className="text-primary text-xs font-bold font-heading">A</span>
                </div>
            )}
            <span className="text-xs font-semibold text-muted-foreground">
                {isUser ? 'Anda' : 'AMANA'}
            </span>
        </div>

        {message.isLoading ? (
          <div className="py-2">
            <LoadingDots />
          </div>
        ) : (
          <>
            <div className="prose prose-sm prose-slate dark:prose-invert max-w-none break-words leading-relaxed
                prose-p:leading-relaxed prose-pre:bg-slate-950 prose-pre:p-4 prose-pre:rounded-lg
                prose-li:marker:text-primary/60 prose-a:text-primary">
              <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
            </div>

            {/* Display Meta Data if available */}
            {message.responseMeta && (
              <div className="mt-6 pt-4 border-t border-border/50 flex flex-wrap gap-x-6 gap-y-2 text-xs text-muted-foreground/80">
                {message.responseMeta.support_info && (
                  <div className="flex items-center gap-1.5" title="Support Level">
                    <ShieldCheck className="w-3.5 h-3.5 text-primary/70" />
                    <span>Level: <strong>{message.responseMeta.support_info.support_level}</strong></span>
                    {message.responseMeta.support_info.support_score && (
                       <span className="ml-1 opacity-70">({(message.responseMeta.support_info.support_score * 100).toFixed(0)}%)</span>
                    )}
                  </div>
                )}

                {message.responseMeta.generation_meta?.generation_mode && (
                  <div className="flex items-center gap-1.5" title="Generation Mode">
                    <Zap className="w-3.5 h-3.5 text-amber-500/70" />
                    <span className="capitalize">{message.responseMeta.generation_meta.generation_mode.replace('_', ' ')}</span>
                  </div>
                )}

                {message.responseMeta.elapsed_seconds && (
                  <div className="flex items-center gap-1.5" title="Response Time">
                    <Clock className="w-3.5 h-3.5 text-blue-500/70" />
                    <span>{message.responseMeta.elapsed_seconds.toFixed(2)}s</span>
                  </div>
                )}
              </div>
            )}

            {/* Layout Sources Accordion */}
            {message.responseMeta?.sources && message.responseMeta.sources.length > 0 && (
                <AmanaSources sources={message.responseMeta.sources} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
