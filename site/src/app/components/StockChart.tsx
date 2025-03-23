'use client';

import React, { useEffect, useState, useRef } from 'react';

interface StockChartProps {
  title: string;
  description?: string;
  chartType?: 'price' | 'macd' | 'rsi' | 'volume';
  ticker?: string;
  height?: number;
}

interface ChartData {
  ticker: string;
  dates: string[];
  [key: string]: any; // For additional data based on chart type
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  date: string;
  value: number;
  secondaryValue?: number;
}

interface LegendItem {
  color: string;
  label: string;
  strokeDasharray?: string;
}

const StockChart: React.FC<StockChartProps> = ({ 
  title, 
  description,
  chartType = 'price',
  ticker = 'SPY',
  height = 300
}) => {
  const [data, setData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    x: 0,
    y: 0,
    date: '',
    value: 0,
  });
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch(`/api/stock-data?type=${chartType}&ticker=${ticker}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch data: ${response.status}`);
        }
        
        const result = await response.json();
        if (result.error) {
          throw new Error(result.error);
        }
        
        setData(result);
        setError(null);
      } catch (err) {
        console.error('Error fetching chart data:', err);
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [chartType, ticker]);

  // Function to get the data array based on chart type
  const getDataArray = (): number[] => {
    if (!data) return [];
    
    switch (chartType) {
      case 'price':
        return data.close || [];
      case 'macd':
        return data.macd || [];
      case 'rsi':
        return data.rsi || [];
      case 'volume':
        return data.volume || [];
      default:
        return [];
    }
  };

  // Function to get y-axis label based on chart type
  const getYAxisLabel = (): string => {
    switch (chartType) {
      case 'price':
        return 'Price ($)';
      case 'macd':
        return 'MACD Value';
      case 'rsi':
        return 'RSI Value';
      case 'volume':
        return 'Volume';
      default:
        return '';
    }
  };

  // Function to get legend items based on chart type
  const getLegendItems = (): LegendItem[] => {
    switch (chartType) {
      case 'price':
        return [
          { color: '#4c6ef5', label: 'Price' },
          { color: '#ff922b', label: '200-day MA' }
        ];
      case 'macd':
        return [
          { color: '#4c6ef5', label: 'MACD Line' },
          { color: '#ff922b', label: 'Signal Line', strokeDasharray: '5,5' }
        ];
      case 'rsi':
        return [
          { color: '#22b8cf', label: 'RSI' },
          { color: '#ff6b6b', label: 'Overbought (70)', strokeDasharray: '5,5' },
          { color: '#51cf66', label: 'Oversold (30)', strokeDasharray: '5,5' }
        ];
      case 'volume':
        return [
          { color: '#4c6ef5', label: 'Volume' }
        ];
      default:
        return [];
    }
  };

  // Function to get secondary data array (if applicable)
  const getSecondaryDataArray = (): number[] => {
    if (!data) return [];
    
    switch (chartType) {
      case 'price':
        return data.ma200 || [];
      case 'macd':
        return data.signal || [];
      default:
        return [];
    }
  };

  // Function to get min/max values for y-axis
  const getYAxisRange = (): { min: number, max: number } => {
    const values = getDataArray();
    const secondaryValues = getSecondaryDataArray();
    
    // If it's RSI, use fixed range of 0-100
    if (chartType === 'rsi') {
      return { min: 0, max: 100 };
    }
    
    if (values.length === 0) {
      return { min: 0, max: 100 };
    }
    
    let allValues = [...values];
    if (secondaryValues.length > 0) {
      allValues = [...allValues, ...secondaryValues];
    }
    
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    
    // Add a small buffer above and below
    const range = max - min;
    const buffer = range * 0.1;
    
    return {
      min: min - buffer,
      max: max + buffer
    };
  };

  // Function to generate SVG path from data
  const generateChartPath = (): string => {
    if (!data || !data.dates || data.dates.length === 0) {
      return '';
    }

    const values = getDataArray();
    if (values.length === 0) return '';

    // Normalize values to fit chart height (0-300)
    const { min, max } = getYAxisRange();
    const range = max - min;
    
    // Generate path points
    const width = 800;
    const height = 300;
    const pointWidth = width / (values.length - 1);
    
    return values.map((value, i) => {
      const x = i * pointWidth;
      // For RSI, use a fixed scale (0-100)
      const y = chartType === 'rsi' 
        ? height * (1 - value / 100)  // Simplified calculation for RSI
        : range === 0 
          ? height / 2 
          : height * (1 - (value - min) / range); // Simplified calculation for other charts
      
      return `${i === 0 ? 'M' : 'L'}${x},${y}`;
    }).join(' ');
  };

  // Function to generate secondary line path (like signal line for MACD or MA for price)
  const generateSecondaryPath = (): string => {
    if (!data || !data.dates || data.dates.length === 0) {
      return '';
    }

    const values = getSecondaryDataArray();
    if (values.length === 0) return '';

    // Get the same normalization as the main line for comparison
    const { min, max } = getYAxisRange();
    const range = max - min;
    
    const width = 800;
    const height = 300;
    const pointWidth = width / (values.length - 1);
    
    return values.map((value, i) => {
      const x = i * pointWidth;
      const y = range === 0 
        ? height / 2 
        : height * (1 - (value - min) / range); // Simplified calculation matching the main chart
      
      return `${i === 0 ? 'M' : 'L'}${x},${y}`;
    }).join(' ');
  };

  // Format tooltip value based on chart type
  const formatTooltipValue = (value: number): string => {
    switch (chartType) {
      case 'price':
        return `$${value.toFixed(2)}`;
      case 'volume':
        return value >= 1000000 
          ? `${(value / 1000000).toFixed(2)}M` 
          : value >= 1000 
            ? `${(value / 1000).toFixed(0)}K` 
            : value.toFixed(0);
      case 'rsi':
        return value.toFixed(1);
      case 'macd':
        return value.toFixed(1);
      default:
        return value.toFixed(2);
    }
  };
  
  // Format y-axis label (slightly different from tooltip for better readability)
  const formatYAxisLabel = (value: number): string => {
    switch (chartType) {
      case 'price':
        return `$${value.toFixed(2)}`;
      case 'volume':
        return value >= 1000000 
          ? `${(value / 1000000).toFixed(1)}M` 
          : value >= 1000 
            ? `${(value / 1000).toFixed(0)}K` 
            : value.toFixed(0);
      case 'rsi':
        return value.toFixed(1);
      case 'macd':
        return value.toFixed(1);
      default:
        return value.toFixed(1);
    }
  };

  // Generate Y-axis labels
  const generateYAxisLabels = () => {
    const { min, max } = getYAxisRange();
    
    if (chartType === 'rsi') {
      // Fixed labels for RSI (0-100 scale) with standard evenly spaced values
      return (
        <>
          <div>100.0</div>
          <div>75.0</div>
          <div>50.0</div>
          <div>25.0</div>
          <div>0.0</div>
        </>
      );
    }
    
    // Dynamic labels for other chart types
    return (
      <>
        <div>{formatYAxisLabel(max)}</div>
        <div>{formatYAxisLabel(max - (max - min) * 0.25)}</div>
        <div>{formatYAxisLabel(max - (max - min) * 0.5)}</div>
        <div>{formatYAxisLabel(max - (max - min) * 0.75)}</div>
        <div>{formatYAxisLabel(min)}</div>
      </>
    );
  };

  // For RSI, generate overbought and oversold lines
  const generateRsiLines = () => {
    if (chartType !== 'rsi' || !data) return null;
    
    const height = 300;
    
    // Position the lines at exactly 70% and 30% of the chart height
    // These positions should align perfectly with the 70.0 and 30.0 labels
    // Using the formula: y = height * (1 - value/100)
    const overboughtY = height * 0.3;  // For value=70, 1-70/100 = 0.3
    const oversoldY = height * 0.7;    // For value=30, 1-30/100 = 0.7
    
    return (
      <>
        <line 
          x1="0" 
          y1={overboughtY} 
          x2="800" 
          y2={overboughtY} 
          stroke="#ff6b6b" 
          strokeWidth="1.5" 
          strokeDasharray="5,5" 
        />
        
        <line 
          x1="0" 
          y1={oversoldY} 
          x2="800" 
          y2={oversoldY} 
          stroke="#51cf66" 
          strokeWidth="1.5" 
          strokeDasharray="5,5" 
        />
      </>
    );
  };

  // Generate chart legend
  const generateLegend = () => {
    const legendItems = getLegendItems();
    return (
      <div className="flex flex-wrap gap-3 mb-2 text-xs text-gray-600 dark:text-gray-400">
        {legendItems.map((item, index) => (
          <div key={index} className="flex items-center">
            <div className="w-4 h-0 mr-2 border-b-2" style={{ 
              borderColor: item.color,
              borderStyle: 'solid',
              borderWidth: '2px',
              borderImage: item.strokeDasharray ? `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="6" height="1"><line x1="0" y1="0" x2="6" y2="0" stroke="${encodeURIComponent(item.color)}" stroke-width="2" stroke-dasharray="${item.strokeDasharray}" /></svg>') 1` : 'none'
            }}></div>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    );
  };

  // Generate date labels
  const generateDateLabels = () => {
    if (!data || !data.dates || data.dates.length === 0) {
      return (
        <>
          <span>Loading...</span>
          <span>Loading...</span>
          <span>Loading...</span>
        </>
      );
    }
    
    const dates = data.dates;
    const startDate = dates[0];
    const midDate = dates[Math.floor(dates.length / 2)];
    const endDate = dates[dates.length - 1];
    
    return (
      <>
        <span>{startDate}</span>
        <span>{midDate}</span>
        <span>{endDate}</span>
      </>
    );
  };

  // Handle mouse movement for tooltip
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || !data || !data.dates || data.dates.length === 0) return;
    
    // Get mouse position relative to SVG
    const svgRect = svgRef.current.getBoundingClientRect();
    const x = e.clientX - svgRect.left;
    const y = e.clientY - svgRect.top;
    
    // Calculate which data point we're over
    const width = svgRect.width;
    const dataIndex = Math.min(
      Math.max(0, Math.floor((x / width) * data.dates.length)),
      data.dates.length - 1
    );
    
    const values = getDataArray();
    const secondaryValues = getSecondaryDataArray();
    
    if (values.length === 0) return;
    
    // Create tooltip data
    setTooltip({
      visible: true,
      x: x,
      y: y,
      date: data.dates[dataIndex],
      value: values[dataIndex],
      secondaryValue: secondaryValues.length > 0 ? secondaryValues[dataIndex] : undefined,
    });
  };

  // Handle mouse leave for tooltip
  const handleMouseLeave = () => {
    setTooltip(prev => ({ ...prev, visible: false }));
  };

  // Get tooltip label based on chart type
  const getTooltipLabels = (): { primary: string, secondary?: string } => {
    switch (chartType) {
      case 'price':
        return { primary: 'Price', secondary: 'MA(200)' };
      case 'macd':
        return { primary: 'MACD', secondary: 'Signal' };
      case 'rsi':
        return { primary: 'RSI' };
      case 'volume':
        return { primary: 'Volume' };
      default:
        return { primary: 'Value' };
    }
  };

  return (
    <div className="financial-card dark:bg-gray-800 rounded-lg shadow-md p-5">
      {/* Title and description aligned with chart */}
      <div className="flex mb-2">
        {/* Empty space to align with y-axis labels */}
        <div className="w-16"></div>
        
        {/* Title and description container */}
        <div className="flex-1">
          <h3 className="text-lg font-semibold">{title}</h3>
          {description && <p className="text-gray-600 dark:text-gray-400 text-sm mt-1">{description}</p>}
        </div>
      </div>
      
      {/* Legend aligned with chart */}
      <div className="flex mb-3">
        {/* Empty space to align with y-axis labels */}
        <div className="w-16"></div>
        
        {/* Legend container */}
        <div className="flex-1">
          {!loading && !error && generateLegend()}
        </div>
      </div>
      
      <div className="flex items-stretch">
        {/* Y-axis labels container - using flexbox justify-between to align with grid lines */}
        <div className="w-16 flex flex-col justify-between text-right pr-2 text-xs text-gray-400" style={{ height: `${height}px` }}>
          {!loading && !error && generateYAxisLabels()}
        </div>

        {/* Main chart container */}
        <div className="flex-1">
          <div 
            className="relative bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden" 
            style={{ height: `${height}px` }}
          >
            {loading ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
              </div>
            ) : error ? (
              <div className="absolute inset-0 flex items-center justify-center text-red-500">
                <p>{error}</p>
              </div>
            ) : (
              <>
                <svg 
                  ref={svgRef}
                  width="100%" 
                  height="100%" 
                  viewBox="0 0 800 300" 
                  preserveAspectRatio="none"
                  onMouseMove={handleMouseMove}
                  onMouseLeave={handleMouseLeave}
                >
                  {/* Grid lines */}
                  <g className="text-gray-300 dark:text-gray-700 stroke-current">
                    {/* Horizontal grid lines - 5 lines that match our label positions */}
                    <line x1="0" y1="0" x2="800" y2="0" strokeWidth="1" />
                    <line x1="0" y1="75" x2="800" y2="75" strokeWidth="1" />
                    <line x1="0" y1="150" x2="800" y2="150" strokeWidth="1" />
                    <line x1="0" y1="225" x2="800" y2="225" strokeWidth="1" />
                    <line x1="0" y1="300" x2="800" y2="300" strokeWidth="1" />
                    
                    {/* Vertical grid lines */}
                    <line x1="100" y1="0" x2="100" y2="300" strokeWidth="1" />
                    <line x1="200" y1="0" x2="200" y2="300" strokeWidth="1" />
                    <line x1="300" y1="0" x2="300" y2="300" strokeWidth="1" />
                    <line x1="400" y1="0" x2="400" y2="300" strokeWidth="1" />
                    <line x1="500" y1="0" x2="500" y2="300" strokeWidth="1" />
                    <line x1="600" y1="0" x2="600" y2="300" strokeWidth="1" />
                    <line x1="700" y1="0" x2="700" y2="300" strokeWidth="1" />
                  </g>
                  
                  {/* RSI specific lines */}
                  {generateRsiLines()}
                  
                  {/* Main chart line */}
                  <path
                    d={generateChartPath()}
                    fill="none"
                    stroke={chartType === 'rsi' ? '#22b8cf' : '#4c6ef5'}
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  
                  {/* Secondary line (MA or Signal) */}
                  <path
                    d={generateSecondaryPath()}
                    fill="none"
                    stroke="#ff922b"
                    strokeWidth="1.5"
                    strokeDasharray={chartType === 'price' ? "0" : "5,5"}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  
                  {/* Price area */}
                  {chartType === 'price' && (
                    <path
                      d={`${generateChartPath()} V300 H0 Z`}
                      fill="url(#chartGradient)"
                      fillOpacity="0.2"
                    />
                  )}
                  
                  {/* Gradient definition */}
                  <defs>
                    <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#4c6ef5" stopOpacity="0.5" />
                      <stop offset="100%" stopColor="#4c6ef5" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                </svg>
                
                {/* Tooltip */}
                {tooltip.visible && (
                  <div 
                    className="absolute pointer-events-none bg-white dark:bg-gray-800 p-2 rounded shadow-lg text-sm z-10"
                    style={{ 
                      left: `${tooltip.x + 10}px`, 
                      top: `${tooltip.y - 40}px`,
                      transform: tooltip.x > 300 ? 'translateX(-100%)' : 'translateX(0)' 
                    }}
                  >
                    <div className="font-medium">{tooltip.date}</div>
                    <div>
                      <span className="font-medium">{getTooltipLabels().primary}:</span> {formatTooltipValue(tooltip.value)}
                    </div>
                    {tooltip.secondaryValue !== undefined && (
                      <div>
                        <span className="font-medium">{getTooltipLabels().secondary}:</span> {formatTooltipValue(tooltip.secondaryValue)}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
          
          <div className="mt-4 flex justify-between text-sm text-gray-600 dark:text-gray-400">
            {loading ? (
              <>
                <span>Loading...</span>
                <span>Loading...</span>
                <span>Loading...</span>
              </>
            ) : !error ? (
              generateDateLabels()
            ) : (
              <>
                <span>Error loading data</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockChart; 