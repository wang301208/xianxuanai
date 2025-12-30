import Link from "next/link";
import { useRouter } from "next/router";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import type { IconDefinition } from "@fortawesome/free-solid-svg-icons";
import tw from "tailwind-styled-components";

interface SidebarItem {
  href: string;
  label: string;
  icon: IconDefinition;
}

interface SidebarProps {
  items: SidebarItem[];
}

const Sidebar: React.FC<SidebarProps> = ({ items }) => {
  const router = useRouter();

  return (
    <SidebarContainer>
      <SidebarHeader>
        <SidebarMark>AS</SidebarMark>
        <div>
          <SidebarTitle>autoSuper</SidebarTitle>
          <SidebarSubtitle>Agent Console</SidebarSubtitle>
        </div>
      </SidebarHeader>
      <SidebarMenu>
        {items.map((item) => {
          const isActive = router.pathname === item.href;
          return (
            <Link key={item.href} href={item.href} legacyBehavior passHref>
              <SidebarLink $active={isActive}>
                <SidebarIcon $active={isActive}>
                  <FontAwesomeIcon icon={item.icon} />
                </SidebarIcon>
                <span>{item.label}</span>
              </SidebarLink>
            </Link>
          );
        })}
      </SidebarMenu>
    </SidebarContainer>
  );
};

export default Sidebar;

const SidebarContainer = tw.aside`
  hidden
  w-64
  flex-col
  border-r
  border-gray-200
  bg-white
  px-6
  py-8
  shadow-lg
  lg:flex
`;

const SidebarHeader = tw.div`
  flex
  items-center
  gap-3
`;

const SidebarMark = tw.span`
  flex
  h-10
  w-10
  items-center
  justify-center
  rounded-xl
  bg-emerald-100
  text-sm
  font-semibold
  text-emerald-600
`;

const SidebarTitle = tw.p`
  text-sm
  font-semibold
  uppercase
  tracking-wide
  text-gray-900
`;

const SidebarSubtitle = tw.p`
  text-xs
  text-gray-500
`;

const SidebarMenu = tw.nav`
  mt-10
  flex
  flex-col
  gap-2
`;

const SidebarLink = tw.a<{ $active: boolean }>`
  flex
  items-center
  gap-3
  rounded-xl
  border
  px-3
  py-2.5
  text-sm
  font-medium
  transition
  ${(p) =>
    p.$active
      ? "border-emerald-200 bg-emerald-50 text-emerald-700 shadow-sm"
      : "border-transparent text-gray-600 hover:border-emerald-200 hover:bg-emerald-50 hover:text-emerald-700"}
`;

const SidebarIcon = tw.span<{ $active: boolean }>`
  flex
  h-8
  w-8
  items-center
  justify-center
  rounded-lg
  ${(p) =>
    p.$active
      ? "bg-white text-emerald-600 shadow-inner"
      : "bg-gray-100 text-gray-500"}
`;

