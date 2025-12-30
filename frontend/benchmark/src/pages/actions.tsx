import Head from "next/head";
import tw from "tailwind-styled-components";

const ActionsPage = () => {
  return (
    <>
      <Head>
        <title>Action Plan - autoSuper Benchmark</title>
      </Head>
      <PageWrapper>
        <HeaderSection>
          <div>
            <PageTitle>Action Plan</PageTitle>
            <PageSubtitle>
              Collate follow-up tasks, dependency checks, and review items so
              the team can progress faster.
            </PageSubtitle>
          </div>
        </HeaderSection>
        <PlaceholderCard>
          <PlaceholderTitle>Feature under construction</PlaceholderTitle>
          <PlaceholderBody>
            The action plan will centralise pending workflow steps, dependency
            health checks, and retrospectives. You will be able to track the
            automation roadmap here or add manual reminders for your team.
          </PlaceholderBody>
        </PlaceholderCard>
      </PageWrapper>
    </>
  );
};

export default ActionsPage;

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

const PlaceholderCard = tw.section`
  rounded-3xl
  border
  border-gray-200
  bg-white
  p-8
  shadow-sm
`;

const PlaceholderTitle = tw.h2`
  text-lg
  font-semibold
  text-gray-900
`;

const PlaceholderBody = tw.p`
  mt-3
  max-w-2xl
  text-sm
  leading-relaxed
  text-gray-600
`;
