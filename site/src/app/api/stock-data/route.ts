import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs/promises';

const execAsync = promisify(exec);

// Helper function to run a Python script and get the output
async function runPythonScript(scriptPath: string, args: string[] = []): Promise<string> {
  const pythonPath = 'python3'; // Adjust if needed
  const command = `${pythonPath} ${scriptPath} ${args.join(' ')}`;
  
  try {
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
  const scriptPath = path.join('/home/rohan/code/finance_analysis/src', 'api_fetch_macd.py');
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
  const scriptPath = path.join('/home/rohan/code/finance_analysis/src', 'api_fetch_rsi.py');
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
  const scriptPath = path.join('/home/rohan/code/finance_analysis/src', 'api_fetch_prices.py');
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
        return NextResponse.json({ error: 'Invalid data type requested' }, { status: 400 });
    }
    
    // Check if the data contains an error
    if (data && data.error) {
      console.error(`Data error for ${dataType} - ${ticker}:`, data.error);
      return NextResponse.json({ error: data.error }, { status: 500 });
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json({ error: 'Failed to fetch data' }, { status: 500 });
  }
} 