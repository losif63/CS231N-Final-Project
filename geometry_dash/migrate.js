import GD from 'gd.js';
import fs from 'fs';
import path from 'path';

const gd = new GD();

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Random delay between min and max seconds
const randDelay = (minSec, maxSec) =>
    sleep((Math.random() * (maxSec - minSec) + minSec) * 1000);

const userCache = new Map();

const seedUserCache = () => {
    if (!fs.existsSync('levels')) return;
    for (const starDir of fs.readdirSync('levels')) {
        const metaPath = path.join('levels', starDir, 'metadata');
        if (!fs.existsSync(metaPath)) continue;
        for (const file of fs.readdirSync(metaPath)) {
            if (!file.endsWith('.json')) continue;
            const data = JSON.parse(fs.readFileSync(path.join(metaPath, file), 'utf8'));
            const { accountID, username } = data.author ?? {};
            if (accountID && username) userCache.set(accountID, { username });
        }
    }
};

const lookupUser = async (accountID, retries = 4) => {
    if (userCache.has(accountID)) return userCache.get(accountID);
    for (let attempt = 0; attempt < retries; attempt++) {
        const user = await gd.users.getByAccountID(accountID);
        if (user) {
            userCache.set(accountID, user);
            return user;
        }
        const backoff = Math.pow(2, attempt) * 30; // 30s, 60s, 120s, 240s
        console.warn(`  Lookup returned null, backing off ${backoff}s (attempt ${attempt + 1}/${retries})`);
        await sleep(backoff * 1000);
    }
    return null;
};

// For unregistered creators (no accountID), the username is embedded in the
// level search response's user list, keyed by player ID. We fetch the level
// by its ID and read _userData directly since gd.js doesn't expose it.
const lookupUserByPlayerID = async (levelID, playerID, retries = 4) => {
    const cacheKey = `pid:${playerID}`;
    if (userCache.has(cacheKey)) return userCache.get(cacheKey);
    for (let attempt = 0; attempt < retries; attempt++) {
        const level = await gd.levels.search({ query: levelID });
        // _userData on a SearchedLevel is the single user entry ['playerID', 'username', 'accountID']
        const username = level?._userData?.[1];
        if (username) {
            const result = { username };
            userCache.set(cacheKey, result);
            return result;
        }
        const backoff = Math.pow(2, attempt) * 5;
        console.warn(`  Player ID lookup returned null, backing off ${backoff}s (attempt ${attempt + 1}/${retries})`);
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
    seedUserCache();
    console.log(`Found ${existingIDs.size} already-processed levels in levels/ (${userCache.size} users cached)`);

    // Collect all pending files upfront for progress tracking
    const pending = [];
    for (const starDir of fs.readdirSync('levels_prev')) {
        const metaPath = path.join('levels_prev', starDir, 'metadata');
        if (!fs.existsSync(metaPath)) continue;
        for (const file of fs.readdirSync(metaPath)) {
            if (file.endsWith('.json')) pending.push({ starDir, file });
        }
    }
    const total = pending.filter(({ file }) => {
        const id = parseInt(file);
        return !existingIDs.has(id);
    }).length;
    console.log(`${total} levels to migrate`);

    let migrated = 0, skipped = 0, failed = 0, requestCount = 0, current = 0;

    for (const { starDir, file } of pending) {
        const metaPath = path.join('levels_prev', starDir, 'metadata');
        const data = JSON.parse(fs.readFileSync(path.join(metaPath, file), 'utf8'));

            if (existingIDs.has(data.id)) {
                skipped++;
                continue;
            }

            current++;
            try {
                const accountID = data.author?.accountID;
                let user = null;
                let fromCache = false;
                if (accountID) {
                    fromCache = userCache.has(accountID);
                    user = await lookupUser(accountID);
                    if (!user) throw new Error('user lookup failed after retries');
                    if (!fromCache) requestCount++;
                } else {
                    const cacheKey = `pid:${data.author.id}`;
                    fromCache = userCache.has(cacheKey);
                    user = await lookupUserByPlayerID(data.id, data.author.id);
                    if (!user) throw new Error('player ID lookup failed after retries');
                    if (!fromCache) requestCount++;
                }

                const enriched = {
                    ...data,
                    author: { ...data.author, username: user?.username ?? null }
                };

                const destDir = path.join('levels', starDir, 'metadata');
                fs.mkdirSync(destDir, { recursive: true });
                fs.writeFileSync(path.join(destDir, file), JSON.stringify(enriched, null, 4));

                existingIDs.add(data.id);
                const cacheTag = fromCache ? ' (cached)' : '';
                console.log(`[${current}/${total}] Migrated level ${data.id} (${data.name}) — username: ${enriched.author.username ?? 'null'}${cacheTag}`);
                migrated++;

                if (!fromCache) {
                    // Every 20 network requests take a longer break (45-90s)
                    if (requestCount % 20 === 0) {
                        console.log('  Taking a longer break...');
                        await randDelay(45, 90);
                    } else if (Math.random() < 0.1) {
                        await randDelay(30, 60);
                    } else {
                        await randDelay(2, 5);
                    }
                }
            } catch (e) {
                console.error(`[${current}/${total}] Failed level ${data.id}: ${e.message}`);
                failed++;
                await randDelay(60, 120);
            }
    }

    console.log(`\nDone. Migrated: ${migrated}, Skipped: ${skipped}, Failed: ${failed}`);
};

migrate().catch(console.error);
