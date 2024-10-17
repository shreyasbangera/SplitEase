import { Noto_Sans, Work_Sans } from 'next/font/google';
import "./globals.css";
import { AuthProvider } from "@/components/AuthProvider";
import Navbar from "@/components/Navbar";

const notoSans = Noto_Sans({
  weight: ['400', '500', '700', '900'],
  subsets: ['latin'],
});

export const metadata = {
  title: "BillSplitr",
  description: "Simplify Group Expense",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={notoSans.className}
      >
      <AuthProvider>
      <Navbar />
      {children}
      </AuthProvider>
      </body>
    </html>
  );
}
