import { useState } from "react";
import tw from "tailwind-styled-components";

export interface MemoryEntry {
  id: string | number;
  type: string;
  summary: string;
  detail?: string;
  created_at?: string;
  importance?: number;
}

interface MemoryItemProps {
  entry: MemoryEntry;
}

const MemoryItem: React.FC<MemoryItemProps> = ({ entry }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Card>
      <Header>
        <Tag>{entry.type}</Tag>
        <Timestamp>
          {entry.created_at
            ? new Date(entry.created_at).toLocaleString()
            : "Recently recorded"}
        </Timestamp>
      </Header>
      <Summary>{entry.summary}</Summary>
      {entry.importance !== undefined && (
        <Meta>
          Importance score - <strong>{entry.importance}</strong>
        </Meta>
      )}
      {entry.detail && (
        <>
          <Divider />
          <ToggleButton
            type="button"
            onClick={() => setIsOpen((previous) => !previous)}
          >
            {isOpen ? "Hide details" : "Show details"}
          </ToggleButton>
          {isOpen && <Detail>{entry.detail}</Detail>}
        </>
      )}
    </Card>
  );
};

export default MemoryItem;

const Card = tw.article`
  rounded-2xl
  border
  border-gray-200
  bg-white
  p-6
  shadow-sm
`;

const Header = tw.div`
  flex
  flex-wrap
  items-center
  justify-between
  gap-2
`;

const Tag = tw.span`
  inline-flex
  items-center
  rounded-full
  bg-emerald-50
  px-3
  py-1
  text-xs
  font-medium
  text-emerald-600
`;

const Timestamp = tw.span`
  text-xs
  text-gray-400
`;

const Summary = tw.p`
  mt-3
  text-sm
  font-medium
  text-gray-800
`;

const Meta = tw.p`
  mt-2
  text-xs
  uppercase
  tracking-wide
  text-gray-500
`;

const Divider = tw.hr`
  my-4
  border-gray-200
`;

const ToggleButton = tw.button`
  text-sm
  font-medium
  text-emerald-600
  transition
  hover:text-emerald-700
`;

const Detail = tw.p`
  mt-3
  text-sm
  leading-relaxed
  text-gray-600
  whitespace-pre-line
`;
