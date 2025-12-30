import tw from "tailwind-styled-components";

import MemoryItem, { MemoryEntry } from "./MemoryItem";

interface MemoryListProps {
  entries: MemoryEntry[];
}

const MemoryList: React.FC<MemoryListProps> = ({ entries }) => {
  if (!entries.length) {
    return (
      <EmptyState>
        <EmptyTitle>No memory entries</EmptyTitle>
        <EmptySubtitle>
          As the agent completes tasks and composes summaries, new memories will
          appear here automatically.
        </EmptySubtitle>
      </EmptyState>
    );
  }

  return (
    <ListWrapper>
      {entries.map((entry) => (
        <MemoryItem key={entry.id} entry={entry} />
      ))}
    </ListWrapper>
  );
};

export default MemoryList;

const ListWrapper = tw.div`
  grid
  gap-6
  lg:grid-cols-2
`;

const EmptyState = tw.div`
  flex
  min-h-[240px]
  flex-col
  items-center
  justify-center
  gap-3
  rounded-3xl
  border
  border-gray-200
  bg-white
  p-10
  text-center
  shadow-sm
`;

const EmptyTitle = tw.h3`
  text-lg
  font-semibold
  text-gray-800
`;

const EmptySubtitle = tw.p`
  max-w-md
  text-sm
  text-gray-500
`;
