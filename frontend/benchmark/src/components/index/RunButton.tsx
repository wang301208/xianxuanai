import React, { useState, useEffect } from "react";

import tw from "tailwind-styled-components";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleNotch } from "@fortawesome/free-solid-svg-icons";

interface RunButtonProps {
  testRun: () => Promise<void>;
  isLoading: boolean;
  cutoff?: string | number | null;
  isMock: boolean;
}

const RunButton: React.FC<RunButtonProps> = ({
  testRun,
  isLoading,
  cutoff,
  isMock,
}) => {
  const intCutoff = cutoff !== undefined && cutoff !== null ? parseInt(String(cutoff)) : null;
  const [timeElapsed, setTimeElapsed] = useState<number>(0);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;

    if (isLoading) {
      interval = setInterval(() => {
        setTimeElapsed((prevTime) => prevTime + 1);
      }, 1000);
    } else {
      if (interval !== null) {
        clearInterval(interval);
      }
      setTimeElapsed(0); // Reset the timer when not loading
    }

    return () => {
      if (interval !== null) {
        clearInterval(interval);
      }
    };
  }, [isLoading]);

  const timeUntilCutoff = intCutoff ? intCutoff - timeElapsed : null;

  const handleClick = () => {
    if (!isLoading) {
      void testRun();
    }
  };

  return (
    <>
      <RunButtonWrapper onClick={handleClick} disabled={isLoading}>
        {!isLoading ? (
          isMock ? "Start mock run" : "Start run"
        ) : (
          <FontAwesomeIcon size="lg" icon={faCircleNotch} spin />
        )}
      </RunButtonWrapper>
      {cutoff && isLoading && (
        <>
          {isMock ? (
            <p>Time elapsed: {timeElapsed} seconds</p>
          ) : (
            <p>Time until cutoff: {timeUntilCutoff} seconds</p>
          )}
        </>
      )}
    </>
  );
};

export default RunButton;

const RunButtonWrapper = tw.button`
  mt-4
  inline-flex
  h-10
  min-w-[140px]
  items-center
  justify-center
  rounded-full
  bg-emerald-400
  px-5
  text-sm
  font-semibold
  text-slate-950
  transition
  hover:bg-emerald-300
  focus:outline-none
  focus:ring
  focus:ring-emerald-400/50
  disabled:cursor-not-allowed
  disabled:opacity-60
`;
