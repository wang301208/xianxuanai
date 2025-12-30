import { PropsWithChildren } from "react";
import {
  faBrain,
  faChartLine,
  faListCheck,
  faProjectDiagram,
} from "@fortawesome/free-solid-svg-icons";
import tw from "tailwind-styled-components";

import Header from "./Header";
import Sidebar from "./Sidebar";

const NAV_ITEMS = [
  { href: "/", label: "Task Graph", icon: faProjectDiagram },
  { href: "/data", label: "Benchmark Results", icon: faChartLine },
  { href: "/memory", label: "Memory Module", icon: faBrain },
  { href: "/actions", label: "Action Plan", icon: faListCheck },
];

const Layout: React.FC<PropsWithChildren> = ({ children }) => {
  return (
    <Shell>
      <Sidebar items={NAV_ITEMS} />
      <MainColumn>
        <Header navItems={NAV_ITEMS} />
        <MainContent>{children}</MainContent>
      </MainColumn>
    </Shell>
  );
};

export default Layout;

const Shell = tw.div`
  min-h-screen
  bg-gray-50
  text-gray-900
  lg:flex
`;

const MainColumn = tw.div`
  flex
  min-h-screen
  flex-1
  flex-col
`;

const MainContent = tw.main`
  flex-1
  overflow-y-auto
  px-6
  py-8
  sm:px-10
  lg:px-12
`;
