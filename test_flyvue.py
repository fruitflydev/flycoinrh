"""FLYVUE smoke tests: at least six assertions for each deliverable module."""
import os,unittest,numpy as np
from PIL import Image
import feed,heatmap,order,run,server,demo
class TestFeed(unittest.TestCase):
 def test_1(self):self.assertEqual(feed.FOV_W,300)
 def test_2(self):self.assertEqual(feed.FOV_H,210)
 def test_3(self):self.assertEqual(feed.MAX_HZ,180.)
 def test_4(self):self.assertEqual(feed.pil_to_luminance(Image.new('RGB',(2,2),'white')).max(),1.)
 def test_5(self):self.assertEqual(feed.fit_to_fov(Image.new('RGB',(100,50))).size,(300,210))
 def test_6(self):self.assertEqual(feed.fit_to_fov(Image.new('RGB',(50,100))).size,(300,210))
 def test_repo_gray_pattern(self):
  if not run.GRAPH.exists() or not run.ANNOTATIONS.exists():self.skipTest('connectome assets absent')
  from flysim import FlyBrain
  from flyeye import FlyEye
  fb=FlyBrain(run.GRAPH); eye=FlyEye(fb,str(run.ANNOTATIONS)); img=Image.new('L',(300,210),128)
  repo=eye.look(np.asarray(img,np.float32)/255.,150,105); got=feed.sample(img,fb,eye,150,105,run.ANNOTATIONS).drive
  self.assertEqual(len(repo),len(got)); self.assertTrue(all(np.allclose(np.asarray(a),np.asarray(b),rtol=1e-6,atol=1e-6) for a,b in zip(repo.values(),got.values())))
class TestHeatmap(unittest.TestCase):
 def test_1(self):self.assertEqual(heatmap.DEFAULT_SIGMA,7.)
 def test_2(self):self.assertEqual(heatmap.fired_density([],np.zeros((2,2)),10,10).shape,(10,10))
 def test_3(self):self.assertGreaterEqual(heatmap.fired_density([0],np.array([[.5,.5],[.1,.1]]),10,10).max(),0)
 def test_4(self):self.assertEqual(heatmap.density_rgba(np.zeros((2,2))).shape,(2,2,4))
 def test_5(self):self.assertEqual(heatmap.overlay_heatmap(Image.new('RGB',(2,2)),np.zeros((2,2))).size,(2,2))
 def test_6(self):self.assertTrue(heatmap.encode_png(Image.new('RGB',(2,2))).startswith(b'\x89PNG'))
class TestOrder(unittest.TestCase):
 def test_1(self):self.assertEqual(order.first_seen(np.zeros((125,3),bool))['1ms']['count'],0)
 def test_2(self):
  x=np.zeros((125,3),bool);x[5,1]=1;self.assertEqual(order.first_seen(x)['5ms']['count'],1)
 def test_3(self):
  x=np.zeros((125,3),bool);x[5,1]=1;self.assertEqual(order.first_seen(x)['10ms']['neurons'],[1])
 def test_4(self):
  x=np.zeros((125,3),bool);x[0,2]=1;self.assertEqual(order.first_seen(x)['1ms']['neurons'],[2])
 def test_5(self):self.assertEqual(len(order.first_seen(np.zeros((125,3),bool))),4)
 def test_6(self):self.assertEqual(order.rank_vector(np.array([[1,0],[0,1]],bool)).tolist(),[0,.2])
class TestRun(unittest.TestCase):
 def test_1(self):self.assertEqual(run.STEPS,125)
 def test_2(self):self.assertEqual(run.DT_MS,.2)
 def test_3(self):self.assertEqual((run.FOV_W,run.FOV_H),(300,210))
 def test_4(self):self.assertTrue(run.MEMORY_CEILING>2_000_000_000)
 def test_5(self):self.assertTrue(callable(run.run_fixation))
 def test_6(self):self.assertTrue(callable(run.run_image))
class TestServer(unittest.TestCase):
 def test_1(self):self.assertEqual(server.PORT,int(os.environ.get('PORT','4660')))
 def test_2(self):self.assertEqual(server.safe_name('../../a b.png'),'a_b.png')
 def test_3(self):self.assertEqual(server.safe_name(''),'upload')
 def test_4(self):self.assertEqual(server.LIMIT,10*1024*1024)
 def test_5(self):self.assertEqual(server.ATTRIBUTION,'Connectome © HHMI Janelia / FlyEM / Google Research, CC-BY')
 def test_6(self):self.assertIsNotNone(server.make_app())
class TestDemo(unittest.TestCase):
 def test_1(self):self.assertEqual(len(demo.EXT),4)
 def test_2(self):self.assertIn('.png',demo.EXT)
 def test_3(self):self.assertIn('.webp',demo.EXT)
 def test_4(self):self.assertTrue(callable(demo.main))
 def test_5(self):self.assertTrue(hasattr(demo,'Path'))
 def test_6(self):self.assertTrue(hasattr(demo,'Image'))
if __name__=='__main__':unittest.main()
