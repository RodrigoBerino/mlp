// src/data/csv_reader.hpp
// Phase 7 — Minimal CSV reader: header + rows of raw strings.

#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp {

// All fields kept as strings; caller converts to T.
struct CsvData {
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;
};

// read_csv: opens 'path', parses comma-separated lines.
// First non-empty line → header.  Remaining non-empty lines → rows.
// Throws std::runtime_error if the file cannot be opened.
inline CsvData read_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("read_csv: cannot open file: " + path);
    }

    CsvData result;
    std::string line;
    bool header_done = false;

    while (std::getline(file, line)) {
        if (line.empty()) { continue; }

        std::vector<std::string> fields;
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }

        if (!header_done) {
            result.header  = std::move(fields);
            header_done = true;
        } else {
            result.rows.push_back(std::move(fields));
        }
    }

    return result;
}

} // namespace mlp
