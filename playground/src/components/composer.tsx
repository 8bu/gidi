import type { KeyboardEvent, Ref } from "react"
import { CornerDownLeftIcon, EraserIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Field, FieldLabel } from "@/components/ui/field"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Spinner } from "@/components/ui/spinner"

type ComposerProps = {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  onClear: () => void
  running: boolean
  /** Run is unavailable (e.g. the browser model is still loading); typing still works. */
  disabled?: boolean
  canClear: boolean
  invalid: boolean
  textareaRef: Ref<HTMLTextAreaElement>
}

export function Composer({
  value,
  onChange,
  onSubmit,
  onClear,
  running,
  disabled = false,
  canClear,
  invalid,
  textareaRef,
}: ComposerProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey) return
    // Enter confirms a Telex/VNI composition; it must not run the note.
    if (event.nativeEvent.isComposing || event.keyCode === 229) return
    event.preventDefault()
    if (!running && !disabled) onSubmit()
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        if (!running && !disabled) onSubmit()
      }}
    >
      <Field data-invalid={invalid || undefined}>
        <FieldLabel htmlFor="note" className="sr-only">
          Note
        </FieldLabel>
        <InputGroup className="has-disabled:bg-transparent has-disabled:opacity-100">
          <InputGroupTextarea
            id="note"
            ref={textareaRef}
            lang="vi"
            value={value}
            rows={3}
            // readOnly (not disabled) keeps focus in the box while a run is in flight.
            readOnly={running}
            aria-busy={running}
            aria-invalid={invalid}
            autoComplete="off"
            autoCapitalize="off"
            spellCheck={false}
            placeholder="Type a note, e.g. mượn chú hai 5 xị"
            className="max-h-48 min-h-20"
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={handleKeyDown}
          />
          <InputGroupAddon align="block-end" className="flex-wrap gap-y-2">
            <span className="hidden items-center gap-3 text-xs sm:flex">
              <KbdGroup>
                <Kbd>
                  <CornerDownLeftIcon />
                </Kbd>
                <span>run</span>
              </KbdGroup>
              <KbdGroup>
                <Kbd>Shift</Kbd>
                <Kbd>
                  <CornerDownLeftIcon />
                </Kbd>
                <span>new line</span>
              </KbdGroup>
            </span>
            <div className="ml-auto flex items-center gap-2">
              <InputGroupButton
                variant="ghost"
                size="sm"
                className="max-sm:h-10 max-sm:px-3"
                disabled={running || !canClear}
                onClick={onClear}
              >
                <EraserIcon data-icon="inline-start" />
                Clear
              </InputGroupButton>
              <Button
                type="submit"
                disabled={running || disabled}
                className="max-sm:h-10 max-sm:px-4"
              >
                {running ? (
                  <Spinner data-icon="inline-start" />
                ) : (
                  <CornerDownLeftIcon data-icon="inline-start" />
                )}
                {running ? "Running" : "Run"}
              </Button>
            </div>
          </InputGroupAddon>
        </InputGroup>
      </Field>
    </form>
  )
}
