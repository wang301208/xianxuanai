import { useEffect, useMemo, useState } from "react";
import Head from "next/head";
import tw from "tailwind-styled-components";

import Graph from "../components/index/Graph";
import TaskInfo from "../components/index/TaskInfo";
import { TaskData } from "../lib/types";

type GraphPayload = {
  nodes?: Array<Record<string, any>>;
  edges?: Array<Record<string, any>>;
  generated_at?: string;
  [key: string]: unknown;
};

const HomePage = () => {
  const [graphData, setGraphData] = useState<GraphPayload | null>(null);
  const [selectedTask, setSelectedTask] = useState<TaskData | null>(null);
  const [isTaskInfoExpanded, setIsTaskInfoExpanded] = useState(false);

  useEffect(() => {
    fetch("/graph.json")
      .then((response) => response.json())
      .then((payload: GraphPayload) => {
        setGraphData(payload);
      })
      .catch((error) => {
        console.error("Error fetching the graph data:", error);
      });
  }, []);

  const totalNodes = useMemo(
    () => graphData?.nodes?.length ?? 0,
    [graphData?.nodes]
  );
  const totalEdges = useMemo(
    () => graphData?.edges?.length ?? 0,
    [graphData?.edges]
  );
  const uniqueCategories = useMemo(() => {
    if (!graphData?.nodes) {
      return 0;
    }
    const categories = new Set<string>();
    graphData.nodes.forEach((node) => {
      const nodeCategories = node?.data?.category;
      if (Array.isArray(nodeCategories)) {
        nodeCategories.forEach((category) =>
          categories.add(String(category).trim())
        );
      } else if (nodeCategories) {
        categories.add(String(nodeCategories).trim());
      }
    });
    return categories.size;
  }, [graphData?.nodes]);

  const isDetailsExpanded = Boolean(isTaskInfoExpanded || selectedTask);

  return (
    <>
      <Head>
        <title>Task Graph - autoSuper Benchmark</title>
      </Head>
      <PageWrapper>
        <SectionHeader>
          <div>
            <SectionTitle>Task Graph</SectionTitle>
            <SectionSubtitle>
              Explore dependencies, trigger benchmark runs, and keep recent
              history within a single workspace.
            </SectionSubtitle>
          </div>
          <MetaNote>
            Last updated - {graphData?.generated_at ?? "just now"}
          </MetaNote>
        </SectionHeader>
        <StatsGrid>
          <StatCard>
            <StatLabel>Tasks</StatLabel>
            <StatValue>{totalNodes}</StatValue>
          </StatCard>
          <StatCard>
            <StatLabel>Dependencies</StatLabel>
            <StatValue>{totalEdges}</StatValue>
          </StatCard>
          <StatCard>
            <StatLabel>Categories</StatLabel>
            <StatValue>{uniqueCategories}</StatValue>
          </StatCard>
        </StatsGrid>
        {!graphData ? (
          <LoadingState>
            <LoadingPulse />
            <LoadingMessage>Loading task graph data...</LoadingMessage>
          </LoadingState>
        ) : (
          <Panels>
            <GraphPanel $isExpanded={isDetailsExpanded}>
              <PanelHeading>
                <div>
                  <PanelTitle>Graph Overview</PanelTitle>
                  <PanelBody>
                    Select a node to inspect details. Use the run console on the
                    right to launch benchmarks instantly.
                  </PanelBody>
                </div>
              </PanelHeading>
              <GraphContainer>
                <Graph
                  graphData={graphData}
                  setSelectedTask={setSelectedTask}
                  setIsTaskInfoExpanded={setIsTaskInfoExpanded}
                />
              </GraphContainer>
            </GraphPanel>
            <DetailsPanel $isExpanded={isDetailsExpanded}>
              <TaskInfo
                selectedTask={selectedTask}
                setIsTaskInfoExpanded={setIsTaskInfoExpanded}
                setSelectedTask={setSelectedTask}
              />
            </DetailsPanel>
          </Panels>
        )}
      </PageWrapper>
    </>
  );
};

export default HomePage;

const PageWrapper = tw.div`
  flex
  flex-col
  gap-8
`;

const SectionHeader = tw.header`
  flex
  flex-wrap
  items-end
  justify-between
  gap-4
`;

const SectionTitle = tw.h1`
  text-2xl
  font-semibold
  text-gray-900
`;

const SectionSubtitle = tw.p`
  mt-1
  max-w-2xl
  text-sm
  text-gray-500
`;

const MetaNote = tw.p`
  text-xs
  text-gray-400
`;

const StatsGrid = tw.div`
  grid
  gap-4
  sm:grid-cols-2
  lg:grid-cols-3
`;

const StatCard = tw.div`
  rounded-2xl
  border
  border-gray-200
  bg-white
  px-5
  py-4
  shadow-sm
`;

const StatLabel = tw.p`
  text-xs
  uppercase
  tracking-wide
  text-gray-500
`;

const StatValue = tw.p`
  mt-2
  text-2xl
  font-semibold
  text-gray-900
`;

const Panels = tw.div`
  flex
  flex-col
  gap-6
  xl:flex-row
`;

const GraphPanel = tw.section<{ $isExpanded: boolean }>`
  flex
  flex-1
  flex-col
  gap-4
  rounded-3xl
  border
  border-gray-200
  bg-white
  p-6
  shadow-sm
  transition-all
  duration-300
  ${(props) => (props.$isExpanded ? "xl:basis-3/5" : "xl:basis-3/4")}
`;

const PanelHeading = tw.div`
  flex
  flex-wrap
  items-center
  justify-between
  gap-3
`;

const PanelTitle = tw.h2`
  text-lg
  font-semibold
  text-gray-900
`;

const PanelBody = tw.p`
  text-sm
  text-gray-500
`;

const GraphContainer = tw.div`
  h-[420px]
  w-full
  xl:h-[520px]
`;

const DetailsPanel = tw.section<{ $isExpanded: boolean }>`
  flex
  w-full
  flex-col
  rounded-3xl
  border
  border-gray-200
  bg-white
  p-6
  shadow-sm
  transition-all
  duration-300
  ${(props) => (props.$isExpanded ? "xl:basis-2/5" : "xl:basis-1/3")}
`;

const LoadingState = tw.div`
  flex
  min-h-[360px]
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
