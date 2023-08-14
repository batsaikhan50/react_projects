const express = require("express");
const cors = require('cors');
const path = require("path");
const https = require("https");
const http = require("http");
const fs = require("fs");

const PORT_HTTPS = 443;
const PORT_HTTP = 80;

const key = fs.readFileSync("./ssl/medsoft.pem");
const cert = fs.readFileSync("./ssl/medsoft.crt");

const app = express();

const httpsServer = https.createServer({key, cert}, app);
const httpServer = http.createServer(app);

app.use(cors({
    origin: "*",
    optionsSuccessStatus: 200
}));
app.use(express.static(path.join(__dirname, "./dist/")));

app.get("/*", (req, res) => {
    res.sendFile(path.join(__dirname, "./dist/index.html"));
});

httpServer.listen(PORT_HTTP, () => {
    console.log(`HTTP SERVER STARTED ON PORT: ${PORT_HTTP}`);
});

httpsServer.listen(PORT_HTTPS, () => {
    console.log(`HTTPS SERVER STARTED ON PORT: ${PORT_HTTPS}`);
});