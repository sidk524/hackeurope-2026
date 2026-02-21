import { Sora } from "next/font/google";
import ProjectsClient from "./components/ProjectsClient";

const sora = Sora({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export default function Home() {
  return <ProjectsClient fontClassName={sora.className} />;
}
