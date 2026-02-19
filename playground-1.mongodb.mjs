// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.
// Import jsontocsv module to convert JSON data to CSV format.
const { jsontocsv } = require("jsontocsv");

// The current database to use.
use('retail_ai');
// Read CSV file and insert data into the collection.

async function importCSV() {
  const csvFilePath = 'C:/Users/i025423/OneDrive - Alliance/Documents/DSTI/instacart-retail-project/data_raw/aisles.csv';  
    const csv = require('csvtojson');
    try {
        const jsonArray = await csv().fromFile(csvFilePath);
        await db.getCollection('aisles').insertMany(jsonArray);
        console.log('CSV data imported successfully!');
    } catch (error) {
        console.error('Error importing CSV data:', error);
    }
}

importCSV();

// Create a new document in the collection.
// db.getCollection('aisles').insertOne({

