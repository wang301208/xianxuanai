import React, { useState } from "react";

import tw from "tailwind-styled-components";

import { TaskData } from "../../lib/types";
import RunButton from "./RunButton";
import MockCheckbox from "./MockCheckbox";

interface SelectedTaskProps {
  selectedTask: TaskData | null;
  isMock: boolean;
  setIsMock: React.Dispatch<React.SetStateAction<boolean>>;
  cutoff: number | null;
  setResponseData: React.Dispatch<React.SetStateAction<any>>;
  allResponseData: any[];
  setAllResponseData: React.Dispatch<React.SetStateAction<any[]>>;
}

const SelectedTask: React.FC<SelectedTaskProps> = ({
  selectedTask,
  isMock,
  setIsMock,
  cutoff,
  setResponseData,
  setAllResponseData,
  allResponseData,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const runTest = async () => {
    // If there's no selected task, do nothing
    if (!selectedTask?.name) return;

    const testParam = selectedTask.name;
    setIsLoading(true);
    try {
      let url = `http://localhost:8000/run_single_test?test=${testParam}&mock=${isMock}`;
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

  return (
    <>
      <TaskName>{selectedTask?.name}</TaskName>
      <TaskPrompt>{selectedTask?.task}</TaskPrompt>
      <MetaGrid>
        <MetaItem>
          <MetaLabel>Cutoff</MetaLabel>
          <MetaValue>{selectedTask?.cutoff}</MetaValue>
        </MetaItem>
        <MetaItem>
          <MetaLabel>Difficulty</MetaLabel>
          <MetaValue>{selectedTask?.info?.difficulty}</MetaValue>
        </MetaItem>
        <MetaItem>
          <MetaLabel>Category</MetaLabel>
          <MetaValue>{selectedTask?.category.join(", ")}</MetaValue>
        </MetaItem>
      </MetaGrid>
      <DetailBlock>
        <MetaLabel>Description</MetaLabel>
        <DetailCopy>{selectedTask?.info?.description}</DetailCopy>
      </DetailBlock>
      <RunButton
        cutoff={selectedTask?.cutoff}
        isLoading={isLoading}
        testRun={runTest}
        isMock={isMock}
      />
      <MockCheckbox isMock={isMock} setIsMock={setIsMock} />
    </>
  );
};

export default SelectedTask;

const TaskName = tw.h1`
  break-words
  text-lg
  font-semibold
  text-gray-900
`;

const TaskPrompt = tw.p`
  break-words
  text-sm
  text-gray-600
`;

const MetaGrid = tw.div`
  mt-4
  grid
  gap-3
  sm:grid-cols-3
`;

const MetaItem = tw.div`
  rounded-lg
  border
  border-emerald-100
  bg-emerald-50
  px-3
  py-2
`;

const MetaLabel = tw.span`
  text-xs
  uppercase
  tracking-wide
  text-emerald-600
`;

const MetaValue = tw.span`
  mt-1
  block
  text-sm
  font-medium
  text-gray-900
`;

const DetailBlock = tw.div`
  mt-4
  flex
  flex-col
  gap-2
`;

const DetailCopy = tw.p`
  text-sm
  text-gray-600
`;
