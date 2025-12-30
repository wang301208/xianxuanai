import React from "react";

import tw from "tailwind-styled-components";

interface MockCheckboxProps {
  isMock: boolean;
  setIsMock: React.Dispatch<React.SetStateAction<boolean>>;
}

const MockCheckbox: React.FC<MockCheckboxProps> = ({ isMock, setIsMock }) => {
  return (
    <CheckboxWrapper>
      <MockCheckboxInput
        type="checkbox"
        checked={isMock}
        onChange={() => setIsMock(!isMock)}
      />
      <span>Run mock test</span>
    </CheckboxWrapper>
  );
};

export default MockCheckbox;

const MockCheckboxInput = tw.input`
  h-4
  w-4
  rounded
  border
  border-gray-300
  bg-white
  text-emerald-500
  focus:outline-none
  focus:ring
  focus:ring-emerald-200
`;

const CheckboxWrapper = tw.label`
  mt-2
  flex 
  items-center 
  gap-3
  text-sm
  text-gray-600
`;
