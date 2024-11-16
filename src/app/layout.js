import { Noto_Sans, Work_Sans } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/components/AuthProvider";
import Navbar from "@/components/Navbar";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/components/theme-provider";

const WorkSans = Work_Sans({
  weight: ["400", "500", "700", "900"],
  subsets: ["latin"],
});

export const metadata = {
  title: "SplitEase",
  description: "Simplify Group Expense",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={WorkSans.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <AuthProvider>
            <Navbar />
            <div className="pt-[54px]">{children}</div>
          </AuthProvider>
        </ThemeProvider>
        <Toaster />
      </body>
    </html>
  );
}
