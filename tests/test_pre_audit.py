import copy
import unittest
import torch
from soflow_fastgen.audit_support import ClassSampler, PhaseFixture, job_seconds, range_penalty


class AuditContracts(unittest.TestCase):
    def test_range_penalty_and_clamp_decomposition(self):
        x = torch.tensor([-2., -.4, .7, 3.], dtype=torch.float64, requires_grad=True)
        g, = torch.autograd.grad(range_penalty(x), x)
        torch.testing.assert_close(g, torch.tensor([-.5,0.,0.,1.],dtype=x.dtype))
        # Non-diagonal Jacobian: blocked image gradients need not correspond
        # to the same fraction of parameter-gradient norm.
        p = torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
        matrix=torch.tensor([[1.,2.],[3.,-1.],[2.,2.]],dtype=p.dtype)
        q=matrix@p+torch.tensor([2.,0.,-3.],dtype=p.dtype)
        y=q.clamp(-1,1); u=torch.tensor([.4,-.7,1.2],dtype=p.dtype)
        actual,=torch.autograd.grad(y,p,u,retain_graph=True)
        blocked,=torch.autograd.grad(q,p,u*(q.abs()>1),retain_graph=True)
        proxy,=torch.autograd.grad(q,p,u)
        torch.testing.assert_close(actual+blocked,proxy)
        eps=1e-6
        direction=torch.tensor([.3,.7],dtype=p.dtype)
        def f(v):return ((matrix@v+torch.tensor([2.,0.,-3.],dtype=v.dtype)).clamp(-1,1)*u).sum()
        torch.testing.assert_close((f(p+eps*direction)-f(p-eps*direction))/(2*eps),(actual*direction).sum())

    def test_phase_and_data_recovery(self):
        phase=PhaseFixture()
        with self.assertRaises(ValueError):phase.rho(1,0)
        self.assertEqual([phase.rho(2,k) for k in range(4)],[0.,.05,.1,.1])
        sampler=ClassSampler([[0,1],[2,3]],71)
        sampler.draw(8); state=copy.deepcopy(sampler.state_dict())
        expected=sampler.draw(32)
        sampler.load_state_dict(state)
        self.assertEqual(expected,sampler.draw(32))

    def test_aggregate_accounting(self):
        jobs=[dict(started=0.,ended=10.),dict(started=5.)]
        self.assertEqual(job_seconds(jobs,15.),20.)
