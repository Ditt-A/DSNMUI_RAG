import { ChatFilters } from "@/types/rag";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface AmanaFiltersProps {
  filters: ChatFilters;
  onChange: (key: keyof ChatFilters, value: any) => void;
  disabled?: boolean;
}

export function AmanaFilters({
  filters,
  onChange,
  disabled,
}: AmanaFiltersProps) {
  return (
    <Card className="p-3 mb-2 flex flex-col md:flex-row md:items-center gap-4 bg-background/50 backdrop-blur shadow-xs border-primary/10">
      <div className="flex-1 min-w-[200px]">
        <Input
          placeholder="Fokus jawaban yang diharapkan..."
          aria-label="Fokus jawaban"
          value={filters.expectedFocus}
          onChange={(e) => onChange("expectedFocus", e.target.value)}
          disabled={disabled}
          className="h-9 bg-background focus-visible:ring-primary/20 bg-white"
        />
      </div>

      <div className="w-full md:w-[220px]">
        <Select
          disabled={disabled}
          value={filters.categoryFilter || "all"}
          onValueChange={(val) =>
            onChange("categoryFilter", val === "all" ? null : val)
          }
        >
          <SelectTrigger className="h-9 bg-white" aria-label="Kategori dokumen">
            <SelectValue placeholder="Kategori" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Semua Kategori</SelectItem>
            <SelectItem value="fatwa">Fatwa</SelectItem>
            <SelectItem value="talimat">Talimat</SelectItem>
            <SelectItem value="pedoman_implementasi">
              Pedoman Implementasi
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-6 text-sm shrink-0 pl-2">
        <div className="flex items-center space-x-2">
          <Switch
            id="use-visual-dalil"
            checked={filters.useVisualDalil}
            onCheckedChange={(val) => onChange("useVisualDalil", val)}
            disabled={disabled}
          />
          <Label
            htmlFor="use-visual-dalil"
            className="cursor-pointer text-muted-foreground"
          >
            Visual Dalil
          </Label>
        </div>
        <div className="flex items-center space-x-2">
          <Switch
            id="use-visual-generator"
            checked={filters.useVisualAsGenerator}
            onCheckedChange={(val) => onChange("useVisualAsGenerator", val)}
            disabled={disabled}
          />
          <Label
            htmlFor="use-visual-generator"
            className="cursor-pointer text-muted-foreground"
          >
            Visual Generator
          </Label>
        </div>
      </div>
    </Card>
  );
}
