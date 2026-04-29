import GD from 'gd.js';
import fs from 'fs';
import path from 'path';

const gd = new GD();

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Random delay between min and max seconds
const randDelay = (minSec, maxSec) =>
    sleep((Math.random() * (maxSec - minSec) + minSec) * 1000);

const lookupUser = async (accountID, retries = 4) => {
    for (let attempt = 0; attempt < retries; attempt++) {
        const user = await gd.users.getByAccountID(accountID);
        if (user) return user;
        const backoff = Math.pow(2, attempt) * 30; // 30s, 60s, 120s, 240s
        console.warn(`  Lookup returned null, backing off ${backoff}s (attempt ${attempt + 1}/${retries})`);
        await sleep(backoff * 1000);
    }
    return null;
};

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

const migrate = async () => {
    const existingIDs = loadExistingIDs();
    console.log(`Found ${existingIDs.size} already-processed levels in levels/`);

    let migrated = 0, skipped = 0, failed = 0, requestCount = 0;

    for (const starDir of fs.readdirSync('levels_prev')) {
        const metaPath = path.join('levels_prev', starDir, 'metadata');
        if (!fs.existsSync(metaPath)) continue;

        for (const file of fs.readdirSync(metaPath)) {
            if (!file.endsWith('.json')) continue;

            const data = JSON.parse(fs.readFileSync(path.join(metaPath, file), 'utf8'));

            if (existingIDs.has(data.id)) {
                skipped++;
                continue;
            }

            try {
                const accountID = data.author?.accountID;
                let user = null;
                if (accountID) {
                    user = await lookupUser(accountID);
                    if (!user) throw new Error('user lookup failed after retries');
                    requestCount++;
                }

                const enriched = {
                    ...data,
                    author: { ...data.author, username: user?.username ?? null }
                };

                const destDir = path.join('levels', starDir, 'metadata');
                fs.mkdirSync(destDir, { recursive: true });
                fs.writeFileSync(path.join(destDir, file), JSON.stringify(enriched, null, 4));

                existingIDs.add(data.id);
                console.log(`Migrated level ${data.id} (${data.name}) — username: ${enriched.author.username ?? 'null'}`);
                migrated++;

                // Every 20 requests take a longer break (45-90s)
                if (requestCount % 20 === 0) {
                    console.log('  Taking a longer break...');
                    await randDelay(45, 90);
                } else {
                    // Base delay: 8-20s, with a 10% chance of a long pause (30-60s)
                    if (Math.random() < 0.1) {
                        await randDelay(30, 60);
                    } else {
                        await randDelay(8, 20);
                    }
                }
            } catch (e) {
                console.error(`Failed level ${data.id}: ${e.message}`);
                failed++;
                // After a failure, always cool down before continuing
                await randDelay(60, 120);
            }
        }
    }

    console.log(`\nDone. Migrated: ${migrated}, Skipped: ${skipped}, Failed: ${failed}`);
};

migrate().catch(console.error);
