package com.example.dataStreamSimulator.services;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvValidationException;

@Service
public class readCSV implements CommandLineRunner{
    private static final Logger logger = LoggerFactory.getLogger(readCSV.class);
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${csv.file.path}")
    private String pathCSV;

    @Value("${csv.file.timeout}")
    private int timeout;

    @Autowired
    private KafkaProducerService kafkaProducerService;

    @Override
    public void run(String... args) throws Exception {
        processPath(pathCSV);
    }

    private void readCsvData(String pathCSV) throws FileNotFoundException, IOException, CsvValidationException{

        try (CSVReader reader = new CSVReaderBuilder(new FileReader(pathCSV)).build()) {
            String [] nextLine;
            String[] headersValues = reader.readNext();
            if (headersValues == null) {
                return;
            }
            long recordCount = 0;
            while ((nextLine = reader.readNext()) != null) {
                ObjectNode jsonvalue = objectMapper.createObjectNode();
                int len = Math.min(headersValues.length, nextLine.length);
                for(int j=0; j<len; j++){
                    jsonvalue.put(headersValues[j], nextLine[j]);
                }
                kafkaProducerService.sendMessage(jsonvalue);
                recordCount++;
                pauseBetweenRecords();
            }
            logger.info("[finished] CSV records sent: " + recordCount + " from " + pathCSV);
        }
    }

    private void readJsonLinesData(String path) throws IOException {
        long recordCount = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            long lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.trim().isEmpty()) {
                    continue;
                }
                JsonNode parsed = objectMapper.readTree(line);
                if (!parsed.isObject()) {
                    throw new IOException("JSONL record must be an object at " + path + ":" + lineNumber);
                }
                kafkaProducerService.sendMessage((ObjectNode) parsed);
                recordCount++;
                pauseBetweenRecords();
            }
        }
        logger.info("[finished] JSONL records sent: " + recordCount + " from " + path);
    }

    private void pauseBetweenRecords() throws IOException {
        if (timeout <= 0) {
            return;
        }
        try {
            Thread.sleep(timeout);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Record streaming interrupted", e);
        }
    }

    private void readData(String path) throws IOException, CsvValidationException {
        String lowerPath = path.toLowerCase();
        if (lowerPath.endsWith(".jsonl") || lowerPath.endsWith(".ndjson")) {
            readJsonLinesData(path);
            return;
        }
        if (lowerPath.endsWith(".csv")) {
            readCsvData(path);
            return;
        }
        throw new IOException("Unsupported input format for " + path + "; expected .csv, .jsonl, or .ndjson");
    }

    private void processPath(String pathCSV) throws IOException, CsvValidationException {
        File source = new File(pathCSV);
        if (source.isFile()) {
            readData(pathCSV);
            return;
        }
        if (source.isDirectory()) {
            File[] files = source.listFiles(File::isFile);
            if (files == null) {
                throw new IOException("Unable to list input directory: " + pathCSV);
            }
            for (File file : files) {
                readData(file.getAbsolutePath());
            }
            return;
        }
        throw new FileNotFoundException("Input path does not exist: " + pathCSV);
    }


}
