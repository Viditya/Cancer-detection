DROP TABLE IF EXISTS tracker;

CREATE TABLE tracker (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ip TEXT NOT NULL,
    search_text TEXT NOT NULL
);
