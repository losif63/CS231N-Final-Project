import GD from 'gd.js';
import fs from 'fs';
import path from 'path';

const gd = new GD();

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const randDelay = (minSec, maxSec) =>
    sleep((Math.random() * (maxSec - minSec) + minSec) * 1000);

const lookupUser = async (accountID, retries = 4) => {
    for (let attempt = 0; attempt < retries; attempt++) {
        const user = await gd.users.getByAccountID(accountID);
        if (user) return user;
        const backoff = Math.pow(2, attempt) * 30;
        console.warn(`  Lookup returned null, backing off ${backoff}s (attempt ${attempt + 1}/${retries})`);
        await sleep(backoff * 1000);
    }
    return null;
};
// const difficulties = ["Auto", "Easy", "Normal", "Hard", "Harder", "Insane", "Easy Demon", "Medium Demon", "Hard Demon", "Insane Demon", "Extreme Demon"];
// const difficulties = ["Easy Demon", "Medium Demon", "Hard Demon", "Insane Demon", "Extreme Demon", "Insane"];
const difficulties = ["Normal", "Hard", "Harder"];
// const lvllengths = ["Tiny", "Short", "Medium", "Long", "XL"];

const loadExistingIDs = () => {
    const ids = new Set();
    if (!fs.existsSync('levels')) return ids;
    for (const starDir of fs.readdirSync('levels')) {
        const metaPath = path.join('levels', starDir, 'metadata');
        if (!fs.existsSync(metaPath)) continue;
        for (const file of fs.readdirSync(metaPath)) {
            if (file.endsWith('.json')) ids.add(parseInt(file));
        }
    }
    return ids;
};

const saveLevelData = async (level, existingIDs) => {
    if (existingIDs.has(level.id)) {
        console.log(`Skipping level ${level.id} (already exists)`);
        return false;
    }
    try {
        const fullLevel = await level.resolve();
        const stars = fullLevel.difficulty.stars || 0;

        const creator = fullLevel.creator.accountID
            ? await lookupUser(fullLevel.creator.accountID)
            : null;
        if (fullLevel.creator.accountID && !creator) throw new Error('user lookup failed after retries');
        const levelData = {
            id: fullLevel.id,
            name: fullLevel.name,
            author: { id: fullLevel.creator.id, accountID: fullLevel.creator.accountID, username: creator?.username ?? null },
            difficulty: {
                stars: stars,
                tier: fullLevel.difficulty.level.pretty,
                requested: fullLevel.difficulty.requestedStars
            },
            stats: {
                objects: fullLevel.stats.objects,
                downloads: fullLevel.stats.downloads,
                likes: fullLevel.stats.likes
            }
        };

        const baseDir = path.join('levels', `${stars}stars`);
        const metaPath = path.join(baseDir, 'metadata');

        fs.mkdirSync(metaPath, { recursive: true });

        fs.writeFileSync(path.join(metaPath, `${fullLevel.id}.json`), JSON.stringify(levelData, null, 4));

        existingIDs.add(fullLevel.id);
        console.log(`Processed level ${fullLevel.id}`);
        return true;
    } catch (e) {
        console.error(`Failed level ${level.id}: ${e.message}`);
        await randDelay(60, 120);
        return false;
    }
};

const runScanner = async () => {
    const existingIDs = loadExistingIDs();
    console.log(`Found ${existingIDs.size} already-processed levels`);
    let requestCount = 0;
    for (const diff of difficulties) {
        console.log(`--- Scanning Difficulty: ${diff} ---`);
        let queryNum = 5000;
        if (diff.includes("Demon")) { queryNum = 1000; }
        const levels = await gd.levels.search({ difficulty: diff }, queryNum);
        console.log(`  Got ${levels.length} levels`);
        for (const lvl of levels) {
            const processed = await saveLevelData(lvl, existingIDs);
            if (processed) {
                requestCount++;
                if (requestCount % 20 === 0) {
                    console.log('  Taking a longer break...');
                    await randDelay(45, 90);
                } else if (Math.random() < 0.1) {
                    await randDelay(30, 60);
                } else {
                    await randDelay(8, 20);
                }
            }
        }
        await randDelay(45, 90); // Cool down between difficulties
    }
};

runScanner().catch(console.error);