import Head from "next/head";
import React, { useEffect, useState } from "react";
import tw from "tailwind-styled-components";

import Dashboard from "~/components/data/Dashboard";
import Reports from "~/components/data/Reports";

const DataPage: React.FC = () => {
  const [data, setData] = useState<any>([]);

  const getData = async () => {
    try {
      const response = await fetch("http://localhost:8000/data");
      if (!response.ok) {
        throw new Error(`Unexpected status ${response.status}`);
      }
      const responseData = await response.json();
      setData(responseData);
    } catch (error) {
      console.error("There was an error fetching the data", error);
    }
  };

  useEffect(() => {
    void getData();
  }, []);

  return (
    <>
      <Head>
        <title>Benchmark Results - autoSuper Benchmark</title>
      </Head>
      <PageWrapper>
        <HeaderSection>
          <div>
            <PageTitle>Benchmark Results</PageTitle>
            <PageSubtitle>
              Review high-level performance metrics and inspect detailed reports
              to identify opportunities for optimisation.
            </PageSubtitle>
          </div>
          <RefreshButton type="button" onClick={() => void getData()}>
            Refresh data
          </RefreshButton>
        </HeaderSection>
        <Card>
          <Dashboard data={data} />
        </Card>
        <Card>
          <Reports data={data} />
        </Card>
      </PageWrapper>
    </>
  );
};

export default DataPage;

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

const RefreshButton = tw.button`
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

const Card = tw.section`
  rounded-3xl
  border
  border-gray-200
  bg-white
  p-6
  shadow-sm
`;
