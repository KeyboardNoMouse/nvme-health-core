import "./globals.css";
export const metadata = { title: "NVMe Health Core", description: "NVMe drive failure predictor & fleet dashboard" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
