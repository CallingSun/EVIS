import torch
import unittest
from src.models.event_encoder import EventContextEncoder
from src.models.fusion_module import MultimodalFusion

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.encoder = EventContextEncoder()
        self.fusion = MultimodalFusion(fusion_type="cross_attention")
        
        # Fixed image and OCR embeddings for testing
        self.v_emb = torch.randn(1, 512)
        self.o_emb = torch.randn(1, 512)
        
        # Fixed event name embedding (simulating CLIP output)
        self.e_name_emb = torch.randn(1, 512)

    def test_distinct_metadata_distinct_fused_embeddings(self):
        """
        Verify that combining the same image with different metadata results in distinct fused embeddings.
        """
        # Metadata 1: Morning in HCMC
        ts1 = "2023-10-27T08:00:00Z"
        lat1, lon1 = 10.762622, 106.660172
        
        # Metadata 2: Night in HCMC (different time)
        ts2 = "2023-10-27T22:00:00Z"
        lat2, lon2 = 10.762622, 106.660172
        
        # Metadata 3: Morning in Hanoi (different location)
        ts3 = "2023-10-27T08:00:00Z"
        lat3, lon3 = 21.028511, 105.804817

        # Encode metadata
        e_emb1 = self.encoder(self.e_name_emb, ts1, lat1, lon1)
        e_emb2 = self.encoder(self.e_name_emb, ts2, lat2, lon2)
        e_emb3 = self.encoder(self.e_name_emb, ts3, lat3, lon3)
        
        # Fuse with SAME image and OCR
        fused1 = self.fusion(self.v_emb, self.o_emb, e_emb1)
        fused2 = self.fusion(self.v_emb, self.o_emb, e_emb2)
        fused3 = self.fusion(self.v_emb, self.o_emb, e_emb3)
        
        # Verify event embeddings are different
        self.assertFalse(torch.allclose(e_emb1, e_emb2), "Event embeddings for different times should be different")
        self.assertFalse(torch.allclose(e_emb1, e_emb3), "Event embeddings for different locations should be different")
        
        # Verify fused embeddings are different
        cos_sim_12 = torch.nn.functional.cosine_similarity(fused1, fused2).item()
        cos_sim_13 = torch.nn.functional.cosine_similarity(fused1, fused3).item()
        
        print(f"Cosine similarity (Time Diff): {cos_sim_12:.6f}")
        print(f"Cosine similarity (Loc Diff): {cos_sim_13:.6f}")
        
        self.assertLess(cos_sim_12, 0.9999, "Fused embeddings for different times should not be identical")
        self.assertLess(cos_sim_13, 0.9999, "Fused embeddings for different locations should not be identical")

    def test_batch_processing(self):
        """Verify that the modules handle batch inputs correctly."""
        batch_size = 4
        v_batch = torch.randn(batch_size, 512)
        o_batch = torch.randn(batch_size, 512)
        e_name_batch = torch.randn(batch_size, 512)
        
        # We need to handle how the encoder processes batches for metadata
        # Current implementation of EventContextEncoder handles one metadata point but repeats it.
        # Let's verify it works for batch embeddings.
        e_batch = self.encoder(e_name_batch, "2023-10-27T10:30:00Z", 10.7, 106.6)
        
        fused_batch = self.fusion(v_batch, o_batch, e_batch)
        self.assertEqual(fused_batch.shape, (batch_size, 512))

if __name__ == "__main__":
    unittest.main()