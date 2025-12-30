import Link from "next/link";
import { useRouter } from "next/router";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faBolt,
  faStopCircle,
  faSyncAlt,
} from "@fortawesome/free-solid-svg-icons";
import tw from "tailwind-styled-components";

import { useAppStatus } from "./AppStatusContext";

interface HeaderProps {
  navItems: Array<{ href: string; label: string }>;
}

const Header: React.FC<HeaderProps> = ({ navItems }) => {
  const { status, refreshStatus, stopCurrentRun } = useAppStatus();
  const router = useRouter();

  return (
    <HeaderBar>
      <HeaderMain>
        <HeaderLeft>
          <BrandMark>AB</BrandMark>
          <BrandCopy>
            <AppTitle>Automated Benchmark Console</AppTitle>
            <AppSubtitle>
              Unified workspace for task graphs, run history, and agent memory.
            </AppSubtitle>
          </BrandCopy>
        </HeaderLeft>
        <HeaderRight>
          <StatusStack>
            <StatusBadge>
              <StatusDot $state={status.agentState} />
              <span>Agent: {status.agentState}</span>
            </StatusBadge>
            <StatusBadge>
              <FontAwesomeIcon icon={faBolt} />
              <span>Learning: {status.learningEnabled ? "ON" : "OFF"}</span>
            </StatusBadge>
            <LastSync>Last sync - {status.lastSync}</LastSync>
          </StatusStack>
          <Actions>
            <ActionButton
              type="button"
              onClick={() => {
                stopCurrentRun();
              }}
            >
              <FontAwesomeIcon icon={faStopCircle} />
              <span>Stop Run</span>
            </ActionButton>
            <PrimaryAction
              type="button"
              onClick={() => {
                refreshStatus();
              }}
            >
              <FontAwesomeIcon icon={faSyncAlt} />
              <span>Refresh Status</span>
            </PrimaryAction>
          </Actions>
        </HeaderRight>
      </HeaderMain>
      <MobileNav>
        {navItems.map((item) => {
          const isActive = router.pathname === item.href;
          return (
            <Link key={item.href} href={item.href} legacyBehavior passHref>
              <MobileNavLink $active={isActive}>{item.label}</MobileNavLink>
            </Link>
          );
        })}
      </MobileNav>
    </HeaderBar>
  );
};

export default Header;

const HeaderBar = tw.header`
  sticky
  top-0
  z-30
  flex
  flex-col
  gap-4
  border-b
  border-gray-200
  bg-white/95
  px-6
  py-4
  shadow-sm
  backdrop-blur
`;

const HeaderMain = tw.div`
  flex
  flex-wrap
  items-center
  justify-between
  gap-6
`;

const HeaderLeft = tw.div`
  flex
  min-w-[260px]
  items-center
  gap-4
`;

const BrandMark = tw.span`
  hidden
  h-12
  w-12
  items-center
  justify-center
  rounded-2xl
  bg-emerald-100
  text-base
  font-semibold
  text-emerald-600
  sm:flex
`;

const BrandCopy = tw.div`
  flex
  flex-col
  gap-1
`;

const AppTitle = tw.h1`
  text-base
  font-semibold
  text-gray-900
  sm:text-lg
`;

const AppSubtitle = tw.p`
  text-xs
  text-gray-500
  sm:text-sm
`;

const HeaderRight = tw.div`
  flex
  flex-wrap
  items-center
  justify-end
  gap-4
`;

const StatusStack = tw.div`
  flex
  flex-wrap
  items-center
  gap-3
`;

const StatusBadge = tw.span`
  inline-flex
  items-center
  gap-2
  rounded-full
  border
  border-gray-200
  bg-gray-50
  px-3
  py-1.5
  text-xs
  font-medium
  text-gray-700
`;

const StatusDot = tw.span<{ $state: string }>`
  h-2.5
  w-2.5
  rounded-full
  ${(p) =>
    p.$state.toLowerCase() === "running" || p.$state.toLowerCase() === "inspecting task"
      ? "bg-emerald-500"
      : p.$state.toLowerCase() === "error"
      ? "bg-red-500"
      : "bg-gray-400"}
`;

const LastSync = tw.span`
  text-xs
  text-gray-500
`;

const Actions = tw.div`
  flex
  items-center
  gap-2
`;

const ActionButton = tw.button`
  inline-flex
  items-center
  gap-2
  rounded-full
  border
  border-gray-200
  bg-white
  px-4
  py-2
  text-xs
  font-medium
  text-gray-700
  transition
  hover:border-gray-300
  hover:bg-gray-50
  focus:outline-none
  focus:ring
  focus:ring-emerald-100
`;

const PrimaryAction = tw.button`
  inline-flex
  items-center
  gap-2
  rounded-full
  border
  border-emerald-200
  bg-emerald-600
  px-4
  py-2
  text-xs
  font-medium
  text-white
  transition
  hover:bg-emerald-700
  focus:outline-none
  focus:ring
  focus:ring-emerald-200
`;

const MobileNav = tw.nav`
  flex
  flex-wrap
  items-center
  gap-2
  lg:hidden
`;

const MobileNavLink = tw.a<{ $active: boolean }>`
  inline-flex
  items-center
  justify-center
  rounded-full
  border
  px-3
  py-1.5
  text-xs
  font-medium
  ${(p) =>
    p.$active
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : "border-gray-200 bg-white text-gray-600 hover:border-emerald-200 hover:bg-emerald-50 hover:text-emerald-700"}
`;
