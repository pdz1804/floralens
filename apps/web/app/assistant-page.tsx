"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { extractCitations, streamAssistant, type AssistantEvent } from "@/lib/api";
import { AlertIcon, BloomIcon, SendIcon, SparkleIcon } from "./icons";

type MessageStatus = "streaming" | "done" | "error";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  status: MessageStatus;
  // Human-readable trace steps (tool calls, delegations) shown while streaming.
  trace: string[];
};

const SUGGESTIONS = [
  "How often should I water a rose?",
  "Does the FloraLens gallery have sunflowers?",
  "What does a healthy orchid leaf look like?",
];

export function AssistantPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const idRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const threadIdRef = useRef(`web-${Date.now().toString(36)}`);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const nextId = () => {
    idRef.current += 1;
    return `m${idRef.current}`;
  };

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || sending) return;
      setError(null);
      setInput("");

      const userMsg: ChatMessage = { id: nextId(), role: "user", text: trimmed, status: "done", trace: [] };
      const assistantId = nextId();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        text: "",
        status: "streaming",
        trace: [],
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setSending(true);

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      const patch = (fn: (m: ChatMessage) => ChatMessage) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

      const onEvent = (event: AssistantEvent) => {
        if (event.type === "answer") {
          patch((m) => ({ ...m, text: event.detail ?? "", status: "done" }));
        } else if (event.type === "error") {
          patch((m) => ({ ...m, text: event.detail ?? "Something went wrong.", status: "error" }));
        } else if (event.type === "limit") {
          patch((m) => ({
            ...m,
            status: m.text ? m.status : "error",
            text: m.text || (event.detail ?? "The assistant stopped before answering."),
          }));
        } else if (event.type === "model" || event.type === "tool") {
          patch((m) => ({ ...m, trace: [...m.trace, event.detail || event.node || event.type] }));
        }
      };

      try {
        await streamAssistant(trimmed, threadIdRef.current, onEvent, ctrl.signal);
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          const message = (e as Error).message;
          setError(message);
          patch((m) => ({ ...m, status: "error", text: m.text || message }));
        }
      } finally {
        setSending(false);
      }
    },
    [sending],
  );

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    void send(input);
  }

  return (
    <div className="assistant" data-testid="assistant-page">
      <section className="assistant-hero">
        <span className="eyebrow">
          <BloomIcon width={14} height={14} /> Naturalist assistant
        </span>
        <h2>Ask about identification, care, or the gallery.</h2>
        <p>
          A multi-agent naturalist — built on the shared AgentForge agent core — searches the web
          for botanical facts (citing its sources) and can delegate gallery-specific questions to a
          care-advisor sub-agent.
        </p>
      </section>

      <div className="assistant-layout card">
        <div className="assistant-messages" ref={scrollRef} data-testid="assistant-messages">
          {messages.length === 0 && (
            <div className="state">
              <span className="state-ico" aria-hidden="true">
                <SparkleIcon width={26} height={26} />
              </span>
              <span className="state-title">Ask the naturalist anything</span>
              <p>Flower identification, care tips, or what&rsquo;s in the FloraLens gallery.</p>
              <div className="assistant-suggestions">
                {SUGGESTIONS.map((s) => (
                  <button
                    type="button"
                    key={s}
                    className="btn btn-ghost assistant-suggestion"
                    onClick={() => void send(s)}
                    disabled={sending}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m) => (
            <div className={`assistant-msg assistant-msg-${m.role}`} key={m.id}>
              <span className="assistant-msg-role">{m.role === "user" ? "You" : "Naturalist"}</span>
              {m.role === "assistant" && m.status === "streaming" && !m.text ? (
                <div className="assistant-thinking" aria-busy="true" aria-label="The naturalist is thinking">
                  <span className="spin" aria-hidden="true" />
                  <span>{m.trace[m.trace.length - 1] ?? "Thinking…"}</span>
                </div>
              ) : (
                <>
                  <p
                    className={`assistant-answer ${m.status === "error" ? "assistant-answer-error" : ""}`}
                    data-testid={m.role === "assistant" ? "assistant-answer" : undefined}
                  >
                    {m.status === "error" && (
                      <AlertIcon width={14} height={14} aria-hidden="true" />
                    )}{" "}
                    {m.text}
                  </p>
                  {m.role === "assistant" && m.status === "done" && extractCitations(m.text).length > 0 && (
                    <ul className="assistant-citations" data-testid="assistant-citations">
                      {extractCitations(m.text).map((url) => (
                        <li key={url}>
                          <a href={url} target="_blank" rel="noopener noreferrer">
                            {url}
                          </a>
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </div>
          ))}
        </div>

        <form className="assistant-composer" onSubmit={onSubmit}>
          <input
            type="text"
            placeholder="Ask about a flower, its care, or the gallery…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            data-testid="assistant-input"
            disabled={sending}
            aria-label="Message the naturalist assistant"
          />
          <button
            type="submit"
            className="btn btn-primary"
            data-testid="assistant-send"
            disabled={sending || !input.trim()}
          >
            {sending ? <span className="spin" aria-hidden="true" /> : <SendIcon width={16} height={16} aria-hidden="true" />}
            Send
          </button>
        </form>
        {error && (
          <p className="err assistant-error" data-testid="error">
            <AlertIcon width={16} height={16} aria-hidden="true" />
            <span>{error}</span>
          </p>
        )}
      </div>
    </div>
  );
}
