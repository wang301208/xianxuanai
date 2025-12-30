import Head from "next/head";
import { useCallback, useEffect, useState } from "react";
import tw from "tailwind-styled-components";

import MemoryList from "../components/memory/MemoryList";
import type { MemoryEntry } from "../components/memory/MemoryItem";

const MemoryPage = () => {
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchMemory = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      if (typeof window !== "undefined") {
        const response = await fetch("/memory");
        if (response.ok) {
          const payload = await response.json();
          const results: MemoryEntry[] = Array.isArray(payload)
            ? payload
            : payload?.results ?? [];
          setMemories(
            results.map((entry, index) => ({
              id: entry.id ?? index,
              type: entry.type ?? "memory",
              summary: entry.summary ?? entry.content ?? "No summary provided.",
              detail: entry.detail ?? entry.content ?? "",
              created_at: entry.created_at ?? entry.timestamp,
              importance: entry.importance,
            }))
          );
          return;
        }
      }
      throw new Error("Memory endpoint not available");
    } catch (error) {
      console.warn("Unable to fetch memory entries, using mock data.", error);
      setErrorMessage(
        "Live memory data is unavailable. Displaying sample entries instead."
      );
      setMemories([
        {
          id: "sample-1",
          type: "Reflection",
          summary: "Agent summarised the strategy after a successful run.",
          detail:
            "Document the critical steps taken to complete the scenario and highlight preparatory context for the next iteration.",
          created_at: new Date().toISOString(),
          importance: 0.8,
        },
        {
          id: "sample-2",
          type: "Context",
          summary: "User requested richer benchmark reports.",
          detail:
            "The next report should include duration, cost breakdown, and traceable links to failing tasks to support audit needs.",
          created_at: new Date().toISOString(),
          importance: 0.6,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchMemory();
  }, [fetchMemory]);

  return (
    <>
      <Head>
        <title>Memory Module - autoSuper Benchmark</title>
      </Head>
      <PageWrapper>
        <HeaderSection>
          <div>
            <PageTitle>Memory Module</PageTitle>
            <PageSubtitle>
              Inspect short-term and long-term agent memories to maintain
              reliable context and reflections.
            </PageSubtitle>
          </div>
          <Actions>
            <SecondaryButton
              type="button"
              onClick={() => {
                void fetchMemory();
              }}
            >
              Refresh
            </SecondaryButton>
          </Actions>
        </HeaderSection>
        {errorMessage && <Alert>{errorMessage}</Alert>}
        {isLoading ? (
          <LoadingState>
            <LoadingPulse />
            <LoadingMessage>Loading memory entries...</LoadingMessage>
          </LoadingState>
        ) : (
          <MemoryList entries={memories} />
        )}
      </PageWrapper>
    </>
  );
};

export default MemoryPage;

const PageWrapper = tw.div`
  flex
  flex-col
  gap-8
`;

const HeaderSection = tw.header`
  flex
  flex-wrap
  items-center
  justify-between
  gap-4
`;

const PageTitle = tw.h1`
  text-2xl
  font-semibold
  text-gray-900
`;

const PageSubtitle = tw.p`
  mt-1
  max-w-2xl
  text-sm
  text-gray-500
`;

const Actions = tw.div`
  flex
  items-center
  gap-2
`;

const SecondaryButton = tw.button`
  inline-flex
  items-center
  justify-center
  rounded-full
  border
  border-gray-200
  bg-white
  px-4
  py-2
  text-sm
  font-medium
  text-gray-700
  transition
  hover:border-gray-300
  hover:bg-gray-50
  focus:outline-none
  focus:ring
  focus:ring-emerald-100
`;

const Alert = tw.div`
  rounded-2xl
  border
  border-amber-200
  bg-amber-50
  px-5
  py-4
  text-sm
  text-amber-700
`;

const LoadingState = tw.div`
  flex
  min-h-[240px]
  flex-col
  items-center
  justify-center
  gap-4
  rounded-3xl
  border
  border-gray-200
  bg-white
  p-10
  shadow-sm
`;

const LoadingPulse = tw.div`
  h-12
  w-12
  animate-spin
  rounded-full
  border-4
  border-gray-200
  border-t-emerald-500
`;

const LoadingMessage = tw.p`
  text-sm
  text-gray-500
`;
