import type { ComponentType } from "react"
import { MonitorIcon, MoonIcon, SunIcon } from "lucide-react"

import { useTheme } from "@/components/theme-provider"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

const OPTIONS: { value: "light" | "dark" | "system"; label: string; icon: ComponentType }[] = [
  { value: "light", label: "Light", icon: SunIcon },
  { value: "dark", label: "Dark", icon: MoonIcon },
  { value: "system", label: "System", icon: MonitorIcon },
]

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size="sm"
      spacing={0}
      value={theme}
      aria-label="Theme"
      // Radix reports "" when the active item is pressed again; keep the choice.
      onValueChange={(value) => {
        if (value === "light" || value === "dark" || value === "system") setTheme(value)
      }}
    >
      {OPTIONS.map(({ value, label, icon: Icon }) => (
        <Tooltip key={value}>
          <TooltipTrigger asChild>
            <ToggleGroupItem value={value} aria-label={`${label} theme`}>
              <Icon />
            </ToggleGroupItem>
          </TooltipTrigger>
          <TooltipContent>{label}</TooltipContent>
        </Tooltip>
      ))}
    </ToggleGroup>
  )
}
