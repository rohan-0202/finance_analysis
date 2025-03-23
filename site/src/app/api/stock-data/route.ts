import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs/promises';

const execAsync = promisify(exec);

// Helper function to run a Python script and get the output
async function runPythonScript(scriptPath: string, args: string[] = []): Promise<string> {
  // Always use python3 explicitly to avoid executable permission issues
  const pythonPath = 'python3'; // Adjust if needed for your environment
  
  // Ensure we're properly passing the script path to the Python interpreter
  // This avoids any issues with executable permissions on the Python files
  const command = `${pythonPath} "${scriptPath}" ${args.join(' ')}`;
  
  try {
    // First check if the script file exists
    try {
      await fs.access(scriptPath);
    } catch (error) {
      console.error(`Python script not found: ${scriptPath}`);
      throw new Error(`Script not found: ${scriptPath}`);
    }
    
    // Now execute the script
    const { stdout, stderr } = await execAsync(command);
    if (stderr) {
      console.error(`Python script error: ${stderr}`);
      // Don't throw if the error is just a warning
      if (stderr.includes('Error') || stderr.includes('Traceback') || stderr.includes('Exception')) {
        throw new Error(stderr);
      }
    }
    return stdout;
  } catch (error) {
    console.error('Failed to execute Python script:', error);
    throw error;
  }
}

// Function to get MACD data for a ticker
async function getMacdData(ticker: string) {
  // Fixed path to point directly to the finance_analysis directory
  const scriptPath = path.resolve('/home/rohan/code/finance_analysis/src', 'api_fetch_macd.py');
  console.log(`Running MACD script at: ${scriptPath}`);
  const output = await runPythonScript(scriptPath, [ticker]);
  
  try {
    return JSON.parse(output);
  } catch (error: any) {
    console.error('Failed to parse JSON from Python script output:', output);
    throw new Error(`Failed to parse data from Python script: ${error.message}`);
  }
}

// Function to get RSI data for a ticker
async function getRsiData(ticker: string) {
  // Fixed path to point directly to the finance_analysis directory
  const scriptPath = path.resolve('/home/rohan/code/finance_analysis/src', 'api_fetch_rsi.py');
  console.log(`Running RSI script at: ${scriptPath}`);
  const output = await runPythonScript(scriptPath, [ticker]);
  
  try {
    return JSON.parse(output);
  } catch (error: any) {
    console.error('Failed to parse JSON from Python script output:', output);
    throw new Error(`Failed to parse data from Python script: ${error.message}`);
  }
}

// Function to get price history for a ticker
async function getPriceHistory(ticker: string) {
  // Fixed path to point directly to the finance_analysis directory
  const scriptPath = path.resolve('/home/rohan/code/finance_analysis/src', 'api_fetch_prices.py');
  console.log(`Running price history script at: ${scriptPath}`);
  const output = await runPythonScript(scriptPath, [ticker]);
  
  try {
    return JSON.parse(output);
  } catch (error: any) {
    console.error('Failed to parse JSON from Python script output:', output);
    throw new Error(`Failed to parse data from Python script: ${error.message}`);
  }
}

// API route handler
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const dataType = searchParams.get('type');
  const ticker = searchParams.get('ticker') || 'SPY';

  try {
    // Validate input parameters
    if (!dataType) {
      return NextResponse.json({ error: 'Missing required parameter: type' }, { status: 400 });
    }
    
    // Attempt to fetch the data
    let data;
    
    switch (dataType) {
      case 'macd':
        data = await getMacdData(ticker);
        break;
      case 'rsi':
        data = await getRsiData(ticker);
        break;
      case 'price':
        data = await getPriceHistory(ticker);
        break;
      case 'volume':
        // Volume uses the same data as price
        data = await getPriceHistory(ticker);
        break;
      default:
        return NextResponse.json({ 
          error: `Invalid data type requested: ${dataType}`,
          validTypes: ['macd', 'rsi', 'price', 'volume']
        }, { status: 400 });
    }
    
    // Check if the data contains an error
    if (data && data.error) {
      console.error(`Data error for ${dataType} - ${ticker}:`, data.error);
      return NextResponse.json({ error: data.error }, { status: 500 });
    }
    
    return NextResponse.json(data);
  } catch (error: any) {
    const errorMessage = error.message || 'Unknown error';
    console.error(`API error for ${dataType} - ${ticker}:`, errorMessage);
    
    return NextResponse.json({ 
      error: 'Failed to fetch data',
      details: errorMessage,
      type: dataType,
      ticker: ticker
    }, { status: 500 });
  }
} 