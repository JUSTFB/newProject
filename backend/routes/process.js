import express from "express";

const router = express.Router();

router.post("/", (req, res) => {
    res.status(200).json({
        ok: true,
        message: "ROUTE HIT SUCCESSFULLY"
    });
});

export default router;
