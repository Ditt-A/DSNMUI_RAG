export interface RagResponse {
  answer: string;
  support_info?: {
    support_level: string;
    support_score: number;
  };
  generation_meta?: {
    generation_mode: string;
  };
  timing?: {
    retrieval_total_ms: number;
  };
  sources?: Array<{
    category: string;
    nomor: string;
    title: string;
    page: number;
    confidence?: string;
    dalil_type?: string;
  }>;
  elapsed_seconds?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  responseMeta?: Omit<RagResponse, 'answer'>;
  isLoading?: boolean;
}

export interface ChatFilters {
  expectedFocus: string;
  categoryFilter: string | null;
  useVisualDalil: boolean;
  useVisualAsGenerator: boolean;
}
