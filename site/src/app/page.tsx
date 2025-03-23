import Image from "next/image";
import Link from "next/link";
import StockChart from "./components/StockChart";

export default function Home() {
  const features = [
    {
      title: "MACD Analysis",
      description: "Moving Average Convergence Divergence analysis tool to identify potential trend reversals and momentum changes in stocks.",
      icon: "📊",
      path: "/macd"
    },
    {
      title: "RSI Indicator",
      description: "Relative Strength Index analysis to identify overbought and oversold conditions in the market.",
      icon: "📈",
      path: "/rsi"
    },
    {
      title: "Technical Graphs",
      description: "Comprehensive technical analysis graphs and charts for in-depth stock market visualization.",
      icon: "📉",
      path: "/technical-graphs"
    },
    {
      title: "Combined Signals",
      description: "Unified view of multiple technical indicators to provide comprehensive trading signals.",
      icon: "🔍",
      path: "/combined-signals"
    }
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-blue-600 to-indigo-900 text-white pt-24 pb-20">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center">
            <div className="md:w-1/2 mb-10 md:mb-0">
              <h1 className="text-4xl md:text-5xl font-bold mb-6">
                Financial Market Analysis Dashboard
              </h1>
              <p className="text-xl mb-8">
                Advanced stock market metrics and visualization tools to help you make data-driven investment decisions.
              </p>
            </div>
            <div className="md:w-1/2 flex justify-center">
              <div className="w-full max-w-md bg-white/10 backdrop-blur-sm rounded-lg p-6 shadow-2xl">
                <div className="h-64 flex items-center justify-center">
                  <svg className="w-48 h-48 text-white/80" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 9H7L10 4L13 14L16 9L18 11L21 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M3 22H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-gray-50 dark:bg-gray-900">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Financial Analysis Tools</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <Link 
                key={index}
                href={feature.path}
                className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600 dark:text-gray-300">{feature.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Sample Charts Section */}
      <section className="py-16 bg-white dark:bg-gray-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Market Analysis Preview</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <StockChart 
              title="S&P 500 Performance" 
              description="Real S&P 500 price with 200-day moving average"
              chartType="price"
              ticker="SPY"
            />
            <StockChart 
              title="MACD Signal Analysis" 
              description="Moving Average Convergence Divergence analysis for S&P 500"
              chartType="macd"
              ticker="SPY"
            />
            <StockChart 
              title="RSI Indicator" 
              description="Relative Strength Index showing overbought and oversold conditions"
              chartType="rsi"
              ticker="SPY"
            />
            <StockChart 
              title="Volume Analysis" 
              description="Trading volume patterns for S&P 500"
              chartType="volume"
              ticker="SPY"
            />
          </div>
        </div>
      </section>

      {/* Simple Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-auto">
        <div className="container mx-auto px-6 text-center">
        </div>
      </footer>
    </div>
  );
}
