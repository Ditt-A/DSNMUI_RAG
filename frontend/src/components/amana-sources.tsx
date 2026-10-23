import { RagResponse } from '@/types/rag';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { BookOpen } from 'lucide-react';

interface AmanaSourcesProps {
  sources: NonNullable<RagResponse['sources']>;
}

export function AmanaSources({ sources }: AmanaSourcesProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <Card className="mt-4 border-primary/10 shadow-xs bg-white overflow-hidden">
      <CardHeader className="py-3 px-4 bg-slate-50/50 border-b border-border/50">
        <CardTitle className="text-sm font-semibold flex items-center gap-2 text-foreground">
          <BookOpen className="w-4 h-4 text-primary" />
          Sumber Referensi
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Accordion className="w-full">
          {sources.map((source, index) => (
            <AccordionItem value={`item-${index}`} key={index} className="border-b-0 px-4">
              <AccordionTrigger className="text-sm py-3 text-left hover:no-underline hover:text-primary transition-colors">
                <span className="font-medium truncate pr-4">
                  {source.nomor} - {source.title}
                </span>
              </AccordionTrigger>
              <AccordionContent className="text-sm text-muted-foreground pb-4 space-y-2">
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 bg-slate-50 rounded-md p-3 border">
                    <div>
                        <span className="block text-xs font-medium text-muted-foreground/70 uppercase">Kategori</span>
                        <span className="text-foreground font-medium">{source.category || '-'}</span>
                    </div>
                    <div>
                        <span className="block text-xs font-medium text-muted-foreground/70 uppercase">Halaman</span>
                        <span className="text-foreground font-medium">{source.page || '-'}</span>
                    </div>
                    {source.confidence && (
                       <div>
                            <span className="block text-xs font-medium text-muted-foreground/70 uppercase">Confidence</span>
                            <span className="text-foreground font-medium">{source.confidence}</span>
                        </div>
                    )}
                    {source.dalil_type && (
                       <div>
                            <span className="block text-xs font-medium text-muted-foreground/70 uppercase">Jenis Dalil</span>
                            <span className="text-foreground font-medium">{source.dalil_type}</span>
                        </div>
                    )}
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </CardContent>
    </Card>
  );
}
