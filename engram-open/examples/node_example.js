const engram = require('../index.node');
console.log("Exported keys:", Object.keys(engram));

if (engram.EngramDb) {
    const db = new engram.EngramDb("./js_test");
    db.store("Test content", { key: "val" });
    console.log("Count:", db.count());
    console.log("Results:", db.recall("Test", 1));
} else {
    console.log("EngramDB not found in exports!");
}
