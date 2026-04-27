import GD from 'gd.js';
import fs from 'fs';
import path from 'path';

const gd = new GD();
const difficulties = ["Auto", "Easy", "Normal", "Hard", "Harder", "Insane", "Easy Demon", "Medium Demon", "Hard Demon", "Insane Demon", "Extreme Demon"];
// const difficulties = ["Easy Demon", "Medium Demon", "Hard Demon", "Insane Demon", "Extreme Demon", "Insane"];
// const difficulties = ["Hard", "Harder"];
const lvllengths = ["Tiny", "Short", "Medium", "Long", "XL"];
const awards = [2, 3];

const saveLevelData = async (level) => {
    try {
        const fullLevel = await level.resolve();
        const stars = fullLevel.difficulty.stars || 0;

        const creator = fullLevel.creator.accountID
            ? await gd.users.getByAccountID(fullLevel.creator.accountID)
            : null;
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

        console.log(`Processed level ${fullLevel.id}`);
        return true;
    } catch (e) {
        console.error(`Failed level ${level.id}: ${e.message}`);
        return false;
    }
};

const runScanner = async () => {
    for (const diff of difficulties) {
        console.log(`--- Scanning Difficulty: ${diff} ---`);
        let queryNum = 5000;
        if (diff.includes("Demon")) { queryNum = 1000; }
        const levels = await gd.levels.search({ difficulty: diff, award: 3 }, queryNum);
        for (const lvl of levels) {
            await saveLevelData(lvl);
            // Random delay 3-7s to mimic human behavior
            await new Promise(r => setTimeout(r, Math.floor(Math.random() * 4000) + 3000));
        }
        await new Promise(r => setTimeout(r, 30000)); // Cool down
    }

};

runScanner().catch(console.error);