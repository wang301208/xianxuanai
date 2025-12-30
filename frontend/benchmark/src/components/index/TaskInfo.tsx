import React, { useState } from "react";

import tw from "tailwind-styled-components";

import { TaskData } from "../../lib/types";
import RunData from "./RunData";
import SelectedTask from "./SelectedTask";
import MockCheckbox from "./MockCheckbox";
import RunButton from "./RunButton";

interface TaskInfoProps {
  selectedTask: TaskData | null;
  setIsTaskInfoExpanded: React.Dispatch<React.SetStateAction<boolean>>;
  setSelectedTask: React.Dispatch<React.SetStateAction<TaskData | null>>;
}

const TaskInfo: React.FC<TaskInfoProps> = ({
  selectedTask,
  setIsTaskInfoExpanded,
  setSelectedTask,
}) => {
  const [isMock, setIsMock] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [allResponseData, setAllResponseData] = useState<any[]>([]);
  const [responseData, setResponseData] = useState<any>();
  const [cutoff, setCutoff] = useState<number | null>(null);

  const runBenchmark = async () => {
    setIsLoading(true);
    try {
      let url = `http://localhost:8000/run?mock=${isMock}`;
      cutoff && !isMock && (url += `&cutoff=${cutoff}`);
      const response = await fetch(url);
      const data = await response.json();

      if (data["returncode"] > 0) {
        throw new Error(data["stderr"]);
      } else {
        const jsonObject = JSON.parse(data["stdout"]);
        setAllResponseData([...allResponseData, jsonObject]);
        setResponseData(jsonObject);
      }
    } catch (error) {
      console.error("There was an error fetching the data", error);
    }
    setIsLoading(false);
  };

  const clearSelection = () => {
    setSelectedTask(null);
    setIsTaskInfoExpanded(false);
  };

  return (
    <TaskDetails>
      <ControlCard>
        <ControlHeader>
          <ControlTitle>Full benchmark</ControlTitle>
          <ControlHint>
            Execute the entire suite and collect updated metrics.
          </ControlHint>
        </ControlHeader>
        <RunButton
          cutoff={selectedTask?.cutoff}
          isLoading={isLoading}
          testRun={runBenchmark}
          isMock={isMock}
        />
        <MockCheckbox isMock={isMock} setIsMock={setIsMock} />
        <ControlFootnote>
          Or choose a node in the graph to trigger a single task run.
        </ControlFootnote>
      </ControlCard>

      {selectedTask && (
        <SelectedTaskCard>
          <SelectedHeader>
            <SelectedLabel>Task selected</SelectedLabel>
            <ClearButton onClick={clearSelection}>Clear selection</ClearButton>
          </SelectedHeader>
          <SelectedTask
            selectedTask={selectedTask}
            isMock={isMock}
            setIsMock={setIsMock}
            cutoff={cutoff}
            setResponseData={setResponseData}
            allResponseData={allResponseData}
            setAllResponseData={setAllResponseData}
          />
        </SelectedTaskCard>
      )}

      {!isMock && (
        <FieldStack>
          <FieldLabel htmlFor="custom-cutoff">Custom cutoff</FieldLabel>
          <CutoffInput
            id="custom-cutoff"
            type="number"
            placeholder="Leave blank for default"
            value={cutoff ?? ""}
            onChange={(event) =>
              setCutoff(event.target.value ? parseInt(event.target.value) : null)
            }
          />
        </FieldStack>
      )}

      <SectionDivider />
      <SectionHeader>Latest run</SectionHeader>
      {!responseData ? (
        <MutedCopy>No runs yet</MutedCopy>
      ) : (
        <RunData latestRun={responseData} />
      )}

      <SectionHeader>Run history</SectionHeader>
      {allResponseData.length <= 1 ? (
        <MutedCopy>No additional runs yet</MutedCopy>
      ) : (
        <HistoryStack>
          {allResponseData.slice(0, -1).map((history, index) => (
            <RunData key={index} latestRun={history} />
          ))}
        </HistoryStack>
      )}
    </TaskDetails>
  );
};

export default TaskInfo;

const TaskDetails = tw.div`
  flex
  flex-col
  gap-6
  rounded-2xl
  border
  border-gray-200
  bg-white
  p-6
  shadow-sm
`;

const ControlCard = tw.div`
  flex
  flex-col
  gap-3
  rounded-xl
  border
  border-gray-200
  bg-gray-50
  p-5
  shadow-sm
`;

const ControlHeader = tw.div`
  flex
  flex-col
  gap-1
`;

const ControlTitle = tw.h3`
  text-sm
  font-semibold
  uppercase
  tracking-wide
  text-gray-700
`;

const ControlHint = tw.p`
  text-xs
  text-gray-500
`;

const ControlFootnote = tw.p`
  text-xs
  text-gray-400
`;

const SelectedTaskCard = tw.div`
  flex
  flex-col
  gap-4
  rounded-xl
  border
  border-emerald-200
  bg-emerald-50
  p-5
`;

const SelectedHeader = tw.div`
  flex
  items-center
  justify-between
  gap-3
`;

const SelectedLabel = tw.span`
  text-xs
  font-semibold
  uppercase
  tracking-wide
  text-emerald-600
`;

const ClearButton = tw.button`
  text-xs
  font-medium
  text-emerald-600
  underline
  transition-colors
  hover:text-emerald-700
`;

const FieldStack = tw.label`
  flex
  flex-col
  gap-2
  text-sm
  text-gray-700
`;

const FieldLabel = tw.span`
  text-xs
  uppercase
  tracking-wide
  text-gray-500
`;

const CutoffInput = tw.input`
  h-9
  rounded-lg
  border
  border-gray-200
  bg-white
  px-3
  text-sm
  text-gray-700
  transition
  focus:border-emerald-400
  focus:outline-none
  focus:ring
  focus:ring-emerald-200
`;

const SectionDivider = tw.hr`
  border-gray-200
`;

const SectionHeader = tw.h4`
  text-sm
  font-semibold
  uppercase
  tracking-wide
  text-gray-600
`;

const MutedCopy = tw.p`
  text-sm
  text-gray-400
`;

const HistoryStack = tw.div`
  flex
  flex-col
  gap-4
`;
