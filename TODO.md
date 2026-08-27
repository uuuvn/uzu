 Reviewed the full diff. Remaining temporary/unfinished items in metal4:

 Actually temporary (must fix before merge):
 4. impl Drop for MetalCommandBufferEncoding { // self.command_encoder.end_encoding(); TODO } — an empty Drop stub; on error paths an Encoding dropped without end_encoding leaves the command buffer in begun state. Implement or delete.

 Sync that was removed and not yet replaced (you said hazard tracker + allocator rewrite is coming, so probably known):
 6. The #[cfg(test)] shared-event watchdog ("prevents tests from freezing, pink screen, shutting down computer") is gone with it — that's the kernel panic you just hit.

 Lost functionality (probably intentional, but flagging):
 7. // TODO: maybe port previous debug encoder labels — the DEBUG_ENCODER_LABELS machinery was dropped.
